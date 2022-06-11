// loop-baker/src/main.rs

use clap::Parser;
use derive_more::Index;
use fxhash::FxHashMap;
use gltf::accessor::Iter;
use gltf::mesh::util::indices::{CastingIter as IndicesCastingIter, U32};
use gltf::mesh::util::tex_coords::{CastingIter as TexCoordsCastingIter, F32};
use gltf::mesh::Mode;
use gltf::{Mesh, Primitive};
use image::{EncodableLayout, ImageBuffer, Pixel, PixelWithColorType, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::{Matrix2, SMatrix, Vector2, Vector3};
use num_traits::Zero;
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use smallvec::SmallVec;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};

const PROGRESS_TICKS_PER_MESH_PRIMITIVE: u64 = 100;

#[derive(Parser)]
struct Args {
    #[clap(help = "Path to the `.gltf` or `.glb` file to be baked")]
    input: PathBuf,
    #[clap(
        short = 's',
        long = "size",
        default_value_t = 2048,
        help = "Horizontal and vertical size of the normal map in texels"
    )]
    size: u32,
    #[clap(
        short = 'c',
        long = "subdivision-count",
        default_value_t = 3,
        help = "Number of subdivisions to perform (higher = slower but higher quality)"
    )]
    subdivision_count: u32,
    #[clap(
        short = 'S',
        long = "object-space",
        help = "Emit an object-space normal map"
    )]
    object_space: bool,
    #[clap(
        short = 'd',
        long = "dump-subdivided-mesh",
        help = "Dump subdivided mesh to `out.obj`"
    )]
    dump_subdivided_mesh: bool,
    #[clap(
        short = 'm',
        long = "margin",
        default_value_t = 10,
        help = "Size of margin in texels, to prevent bleed at seams"
    )]
    margin: u32,
}

struct Texture<P>
where
    P: Pixel + PixelWithColorType,
    [P::Subpixel]: EncodableLayout,
{
    texels: ImageBuffer<P, Vec<P::Subpixel>>,
}

impl<P> Texture<P>
where
    P: Pixel + PixelWithColorType,
    [P::Subpixel]: EncodableLayout,
{
    fn new(size: u32) -> Texture<P> {
        Texture {
            texels: ImageBuffer::new(size, size),
        }
    }

    fn put_pixel(&mut self, pos: Vector2<u32>, color: P) {
        self.texels.put_pixel(pos.x, pos.y, color)
    }

    fn write_to(&self, path: &Path) {
        self.texels.save(path).unwrap();
    }

    fn inflate_margin(&mut self) {
        // TODO(pcwalton): This is really slow!
        let mut dest = ImageBuffer::new(self.texels.width(), self.texels.height());
        for y in 0..(self.texels.height() as i32) {
            for x in 0..(self.texels.width() as i32) {
                let dest_texel = self.texels.get_pixel(x as u32, y as u32);
                let mut color = *dest_texel;
                if dest_texel.is_black() {
                    'outer: for offset_y in (-1)..=1 {
                        for offset_x in (-1)..=1 {
                            if offset_y == 0 && offset_x == 0 {
                                continue;
                            }
                            if let (Ok(dest_x), Ok(dest_y)) =
                                ((x + offset_x).try_into(), (y + offset_y).try_into())
                            {
                                if let Some(src_texel) =
                                    self.texels.get_pixel_checked(dest_x, dest_y)
                                {
                                    if !src_texel.is_black() {
                                        color = *src_texel;
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
                dest.put_pixel(x as u32, y as u32, color);
            }
        }

        mem::swap(&mut dest, &mut self.texels);
    }
}

impl Texture<Rgb<u8>> {
    fn put_unit_vector(&mut self, pos: Vector2<u32>, vector: Vector3<f32>) {
        self.put_pixel(
            pos,
            Rgb([
                ((vector.x + 1.0) * 0.5 * 255.0).round() as u8,
                ((vector.y + 1.0) * 0.5 * 255.0).round() as u8,
                ((vector.z + 1.0) * 0.5 * 255.0).round() as u8,
            ]),
        )
    }
}

trait PixelExt {
    fn is_black(&self) -> bool;
}

impl<P> PixelExt for P
where
    P: Pixel,
{
    fn is_black(&self) -> bool {
        let mut is_black = true;
        (*self).clone().apply(|subpixel| {
            is_black = is_black && subpixel.is_zero();
            subpixel
        });
        is_black
    }
}

struct FaceInfo {
    original_vertex_indices: [usize; 3],
    original_normals: [Vector3<f32>; 3],
    original_tangents: [Vector3<f32>; 3],
    split_lambdas: [Vector3<f32>; 3],
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Vertex {
    provenance: VertexProvenance,
    position: Vector3<f32>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum VertexProvenance {
    Original(u32),
    SplitEdge { from: NodeIndex, to: NodeIndex },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Index)]
struct Face {
    #[index]
    nodes: [NodeIndex; 3],
    original_face_index: usize,
}

impl Face {
    fn new(nodes: [NodeIndex; 3], original_face_index: usize) -> Self {
        Self {
            nodes,
            original_face_index,
        }
    }

    fn split(&self, nodes: [NodeIndex; 3]) -> Face {
        Face::new(nodes, self.original_face_index)
    }
}

#[derive(Clone, Copy)]
struct Derivatives {
    // NB: Not normalized.
    normal: Vector3<f32>,
}

fn bary_interp<T>(a: T, b: T, c: T, lambda: Vector3<f32>) -> T
where
    T: Add<Output = T> + Mul<f32, Output = T>,
{
    a * lambda.x + b * lambda.y + c * lambda.z
}

fn eval_bezier_triangle_2(
    p002: Vector3<f32>,
    p011: Vector3<f32>,
    p020: Vector3<f32>,
    p101: Vector3<f32>,
    p110: Vector3<f32>,
    p200: Vector3<f32>,
    lambda: Vector3<f32>,
) -> Vector3<f32> {
    let q010: Vector3<f32> = bary_interp(p110.into(), p020.into(), p011.into(), lambda);
    let q001: Vector3<f32> = bary_interp(p101.into(), p011.into(), p002.into(), lambda);
    let q100: Vector3<f32> = bary_interp(p200.into(), p110.into(), p101.into(), lambda);

    bary_interp(q100, q010, q001, lambda)
}

fn eval_bezier_triangle_3(
    p003: Vector3<f32>,
    p012: Vector3<f32>,
    p021: Vector3<f32>,
    p030: Vector3<f32>,
    p102: Vector3<f32>,
    p111: Vector3<f32>,
    p120: Vector3<f32>,
    p201: Vector3<f32>,
    p210: Vector3<f32>,
    p300: Vector3<f32>,
    lambda: Vector3<f32>,
) -> Vector3<f32> {
    let q020: Vector3<f32> = bary_interp(p120.into(), p030.into(), p021.into(), lambda);
    let q011: Vector3<f32> = bary_interp(p111.into(), p021.into(), p012.into(), lambda);
    let q110: Vector3<f32> = bary_interp(p210.into(), p120.into(), p111.into(), lambda);
    let q002: Vector3<f32> = bary_interp(p102.into(), p012.into(), p003.into(), lambda);
    let q101: Vector3<f32> = bary_interp(p201.into(), p111.into(), p102.into(), lambda);
    let q200: Vector3<f32> = bary_interp(p300.into(), p210.into(), p201.into(), lambda);

    eval_bezier_triangle_2(q002, q011, q020, q101, q110, q200, lambda)
}

fn eval_bezier_triangle_matrix_3(cp: &SMatrix<f32, 3, 10>, lambda: Vector3<f32>) -> Vector3<f32> {
    let p003 = cp.column(0).into();
    let p012 = cp.column(1).into();
    let p021 = cp.column(2).into();
    let p030 = cp.column(3).into();
    let p102 = cp.column(4).into();
    let p111 = cp.column(5).into();
    let p120 = cp.column(6).into();
    let p201 = cp.column(7).into();
    let p210 = cp.column(8).into();
    let p300 = cp.column(9).into();

    eval_bezier_triangle_3(
        p003, p012, p021, p030, p102, p111, p120, p201, p210, p300, lambda,
    )
}

#[derive(Clone)]
struct MeshGraph {
    graph: UnGraph<Vertex, (), u32>,
    faces: Vec<Face>,
}

impl MeshGraph {
    fn new(graph: UnGraph<Vertex, (), u32>, faces: Vec<Face>) -> MeshGraph {
        MeshGraph { graph, faces }
    }

    fn face_is_regular(&self, face: &Face) -> bool {
        self.graph.neighbors(face[0]).count() == 6
            && self.graph.neighbors(face[1]).count() == 6
            && self.graph.neighbors(face[2]).count() == 6
    }

    fn limit_surface_derivatives(&self, face: &Face, lambda: Vector3<f32>) -> Option<Derivatives> {
        #[rustfmt::skip]
        static M_D_LAMBDA_1: Lazy<SMatrix<f32, 12, 10>> = Lazy::new(|| {
            let m_d_lambda_1: SMatrix<f32, 12, 10> = SMatrix::from_row_slice(&[
                -1.0, -2.0, -3.0, -2.0, -2.0, -4.0, -4.0, -2.0, -4.0,  0.0,
                 1.0,  2.0,  2.0,  0.0,  2.0,  4.0,  4.0,  3.0,  4.0,  2.0,
                 0.0, -2.0, -2.0, -1.0,  2.0,  0.0, -1.0,  2.0,  1.0,  1.0,
                 2.0,  3.0,  2.0,  1.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0,
                 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                -2.0, -1.0,  0.0,  0.0, -3.0, -1.0,  0.0, -2.0, -1.0, -1.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -2.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.0,
                 0.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  1.0,  1.0,
                 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,
                 0.0,  0.0,  1.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            ]);
            m_d_lambda_1.scale(1.0 / 6.0)
        });

        #[rustfmt::skip]
        static M_D_LAMBDA_2: Lazy<SMatrix<f32, 12, 10>> = Lazy::new(|| {
            let m_d_lambda_2: SMatrix<f32, 12, 10> = SMatrix::from_row_slice(&[
                -2.0, -3.0, -2.0, -1.0, -4.0, -4.0, -2.0, -4.0, -2.0,  0.0,
                -1.0, -2.0, -2.0,  0.0, -1.0,  0.0,  2.0,  1.0,  2.0,  1.0,
                 0.0,  2.0,  2.0,  1.0,  4.0,  4.0,  2.0,  4.0,  3.0,  2.0,
                 1.0,  2.0,  3.0,  2.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0,
                 2.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                 1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                -1.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  1.0,  0.0,  1.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -2.0,
                 0.0,  0.0, -1.0, -2.0,  0.0, -1.0, -3.0, -1.0, -2.0, -1.0,
                 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            ]);
            m_d_lambda_2.scale(1.0 / 6.0)
        });

        if !self.face_is_regular(face) {
            return None;
        }

        //       9---8
        //      / \ / \
        //    10---1---7
        //    / \ /X\ / \
        //  11---2---3---6
        //    \ / \ / \ /
        //    12---4---5

        let (v1, v2, v3) = (face[0], face[1], face[2]);

        // These should always succeed for manifold meshes, but alas, not all meshes in the wild
        // are manifold.

        let v10 = self.node_across_from(v1, v2, v3)?;
        let v4 = self.node_across_from(v2, v3, v1)?;
        let v7 = self.node_across_from(v3, v1, v2)?;

        let v9 = self.node_across_from(v1, v10, v2)?;
        let v8 = self.node_across_from(v1, v7, v3)?;
        let v11 = self.node_across_from(v2, v10, v1)?;
        let v12 = self.node_across_from(v2, v4, v3)?;
        let v6 = self.node_across_from(v3, v7, v1)?;
        let v5 = self.node_across_from(v3, v4, v2)?;

        let x: SMatrix<f32, 3, 12> = SMatrix::from_columns(&[
            self.graph[v1].position,
            self.graph[v2].position,
            self.graph[v3].position,
            self.graph[v4].position,
            self.graph[v5].position,
            self.graph[v6].position,
            self.graph[v7].position,
            self.graph[v8].position,
            self.graph[v9].position,
            self.graph[v10].position,
            self.graph[v11].position,
            self.graph[v12].position,
        ]);

        // Use generalized de Casteljau subdivision to evaluate the normal for numerical stability.
        let (d_lambda1_cp, d_lambda2_cp) = (x * *M_D_LAMBDA_1, x * *M_D_LAMBDA_2);
        let d_lambda1 = eval_bezier_triangle_matrix_3(&d_lambda1_cp, lambda);
        let d_lambda2 = eval_bezier_triangle_matrix_3(&d_lambda2_cp, lambda);

        let normal = d_lambda1.cross(&d_lambda2);

        Some(Derivatives { normal })
    }

    fn node_across_from(
        &self,
        edge_v0: NodeIndex,
        edge_v1: NodeIndex,
        across_from: NodeIndex,
    ) -> Option<NodeIndex> {
        self.graph
            .neighbors(edge_v0)
            .find(|v| self.graph.neighbors(edge_v1).any(|node| node == *v) && *v != across_from)
    }

    // Loop subdivision.
    fn subdivide(&mut self) {
        // Move the original vertices. (We will reconstruct all the edges later.)
        let mut new_graph = UnGraph::new_undirected();
        for (node_index, vertex) in self.graph.node_references() {
            let mut new_position = Vector3::zeros();

            let n = self.graph.neighbors(node_index).count();
            for neighbor in self.graph.neighbors(node_index) {
                new_position += self.graph[neighbor].position;
            }
            let u = if n == 3 {
                3.0 / 16.0
            } else {
                3.0 / (8.0 * n as f32)
            };
            new_position = new_position * u + (1.0 - n as f32 * u) * vertex.position;

            let new_node_index = new_graph.add_node(Vertex {
                provenance: vertex.provenance,
                position: new_position,
            });
            assert_eq!(node_index, new_node_index);
        }

        // For each triangle, create new subdivision vertices.
        let mut subdivided_edges = FxHashMap::default();
        for face in &self.faces {
            let (a, b, c) = (face[0], face[1], face[2]);
            let (ab_edge, bc_edge, ca_edge) = self.face_edges(face);

            subdivided_edges
                .entry(ab_edge)
                .or_insert_with(|| self.subdivide_edge(&mut new_graph, a, b, c));
            subdivided_edges
                .entry(bc_edge)
                .or_insert_with(|| self.subdivide_edge(&mut new_graph, b, c, a));
            subdivided_edges
                .entry(ca_edge)
                .or_insert_with(|| self.subdivide_edge(&mut new_graph, c, a, b));
        }

        // Build new split edges in the new graph.
        for edge in self.graph.edge_references() {
            match subdivided_edges.get(&edge.id()) {
                Some(&new_vertex) => {
                    new_graph.update_edge(edge.source(), new_vertex, ());
                    new_graph.update_edge(new_vertex, edge.target(), ());
                }
                None => {
                    new_graph.update_edge(edge.source(), edge.target(), ());
                }
            }
        }

        // Split faces in the new graph.
        //
        //         a
        //        / \
        //      ab---ca
        //      /\   /\
        //     /  \ /  \
        //    b----bc---c
        let mut new_faces = vec![];
        for face in &self.faces {
            let (a, b, c) = (face[0], face[1], face[2]);
            let (ab_edge, bc_edge, ca_edge) = self.face_edges(face);

            let ab = subdivided_edges[&ab_edge];
            let bc = subdivided_edges[&bc_edge];
            let ca = subdivided_edges[&ca_edge];

            new_graph.update_edge(ab, ca, ());
            new_graph.update_edge(ca, bc, ());
            new_graph.update_edge(bc, ab, ());

            new_faces.extend_from_slice(&[
                face.split([a, ca, ab]),
                face.split([b, ab, bc]),
                face.split([c, bc, ca]),
                face.split([ab, ca, bc]),
            ]);
        }
        new_faces.sort();

        // Done.
        self.graph = new_graph;
        self.faces = new_faces;
    }

    fn subdivide_edge(
        &self,
        new_graph: &mut UnGraph<Vertex, (), u32>,
        edge_v0: NodeIndex,
        edge_v1: NodeIndex,
        other_v: NodeIndex,
    ) -> NodeIndex {
        let edge_v0_pos = self.graph[edge_v0].position;
        let edge_v1_pos = self.graph[edge_v1].position;
        let other_v_pos = self.graph[other_v].position;
        let across_v_pos = match self.node_across_from(edge_v0, edge_v1, other_v) {
            Some(across_v) => self.graph[across_v].position,
            None => {
                // This is the edge of a neighborhood and shouldn't really matter, but it's simpler
                // to just return something reasonable. Let's just reflect the vertex across the
                // edge.
                edge_v1_pos - other_v_pos + edge_v0_pos
            }
        };

        let new_pos = ((edge_v0_pos + edge_v1_pos) * 3.0 + other_v_pos + across_v_pos) / 8.0;

        new_graph.add_node(Vertex {
            provenance: VertexProvenance::SplitEdge {
                from: edge_v0,
                to: edge_v1,
            },
            position: new_pos,
        })
    }

    fn face_edges(&self, face: &Face) -> (EdgeIndex, EdgeIndex, EdgeIndex) {
        let (a, b, c) = (face[0], face[1], face[2]);
        let ab_edge = self
            .graph
            .edges_connecting(a, b)
            .next()
            .expect("Face has no edge ab!")
            .id();
        let bc_edge = self
            .graph
            .edges_connecting(b, c)
            .next()
            .expect("Face has no edge bc!")
            .id();
        let ca_edge = self
            .graph
            .edges_connecting(c, a)
            .next()
            .expect("Face has no edge ca!")
            .id();

        (ab_edge, bc_edge, ca_edge)
    }

    #[allow(dead_code)]
    fn dump(&self) {
        let mut file = BufWriter::new(File::create("out.obj").unwrap());

        for (_, vertex) in self.graph.node_references() {
            writeln!(
                &mut file,
                "v {} {} {}",
                vertex.position.x, vertex.position.y, vertex.position.z
            )
            .unwrap();
        }

        for face in &self.faces {
            writeln!(
                &mut file,
                "f {} {} {}",
                face[0].index() + 1,
                face[1].index() + 1,
                face[2].index() + 1
            )
            .unwrap();
        }
    }
}

struct MeshData {
    positions: Vec<Vector3<f32>>,
    normals: Vec<Vector3<f32>>,
    tex_coords: Vec<Vector2<f32>>,
    indices: Vec<u32>,
}

struct NormalMap {
    texture: Texture<Rgb<u8>>,
    name: String,
}

impl NormalMap {
    fn new(name: String, size: u32) -> Self {
        Self {
            texture: Texture::new(size),
            name,
        }
    }
}

struct PrimitiveProcessor<'a> {
    normal_map: &'a mut NormalMap,
    original_data: MeshData,
    graph: MeshGraph,
    duplicates: FxHashMap<NodeIndex, SmallVec<[NodeIndex; 2]>>,
    refined_normals: Vec<Vector3<f32>>,
    original_tangents: Vec<Vector3<f32>>,
}

impl<'a> PrimitiveProcessor<'a> {
    fn new(
        normal_map: &'a mut NormalMap,
        positions: Iter<[f32; 3]>,
        normals: Iter<[f32; 3]>,
        tex_coords: TexCoordsCastingIter<F32>,
        indices: IndicesCastingIter<U32>,
    ) -> Self {
        let positions: Vec<_> = positions
            .map(|slice| Vector3::from_column_slice(&slice))
            .collect();
        let normals = normals
            .map(|slice| Vector3::from_column_slice(&slice))
            .collect();
        let tex_coords = tex_coords
            .map(|slice| Vector2::from_column_slice(&slice))
            .collect();
        let indices: Vec<_> = indices.collect();

        // Build a graph.
        let mut graph = UnGraph::new_undirected();
        let mut faces = Vec::with_capacity(indices.len() / 3);
        for i in 0..positions.len() {
            let node_index = graph.add_node(Vertex {
                provenance: VertexProvenance::Original(i as u32),
                position: positions[i],
            });
            assert_eq!(node_index.index(), i);
        }
        for i in 0..(indices.len() / 3) {
            let a = NodeIndex::new(indices[i * 3 + 0] as usize);
            let b = NodeIndex::new(indices[i * 3 + 1] as usize);
            let c = NodeIndex::new(indices[i * 3 + 2] as usize);
            graph.update_edge(a, b, ());
            graph.update_edge(b, c, ());
            graph.update_edge(c, a, ());
            faces.push(Face::new([a, b, c], i));
        }
        faces.sort();

        Self {
            normal_map,
            original_data: MeshData {
                positions,
                normals,
                tex_coords,
                indices,
            },
            original_tangents: vec![],
            graph: MeshGraph::new(graph, faces),
            duplicates: FxHashMap::default(),
            refined_normals: vec![],
        }
    }

    fn merge_duplicates(&mut self) {
        // Find duplicates.
        let mut positions = FxHashMap::default();
        for (node_index, vertex) in self.graph.graph.node_references() {
            let list = positions
                .entry(to_ordered_float(vertex.position))
                .or_insert(vec![]);
            list.push(node_index);
        }

        // Munge graph.
        //
        // NB: We can't actually remove duplicate vertices because that would shift indices around.
        // Instead we disconnect them.
        for indices in positions.values() {
            if indices.len() == 1 {
                continue;
            }

            let mut indices: SmallVec<[NodeIndex; 2]> = indices.iter().cloned().collect();
            indices.sort();

            let representative = indices[0];
            for &index in &indices[1..] {
                while let Some((neighbor_edge, neighbor)) = self
                    .graph
                    .graph
                    .neighbors(index)
                    .detach()
                    .next(&self.graph.graph)
                {
                    self.graph.graph.remove_edge(neighbor_edge);
                    self.graph.graph.update_edge(neighbor, representative, ());
                }
            }

            self.duplicates.insert(representative, indices);
        }

        // Munge faces.
        for face in &mut self.graph.faces {
            for index in &mut face.nodes {
                let indices = &positions[&to_ordered_float(self.graph.graph[*index].position)];
                if indices.len() == 1 {
                    continue;
                }
                *index = indices.iter().cloned().min().unwrap();
            }
        }

        fn to_ordered_float(vector: Vector3<f32>) -> [OrderedFloat<f32>; 3] {
            [
                OrderedFloat(vector[0]),
                OrderedFloat(vector[1]),
                OrderedFloat(vector[2]),
            ]
        }
    }

    fn compute_original_tangents(&mut self) {
        let mut tangents: Vec<Vector3<f32>> =
            vec![Vector3::zeros(); self.original_data.positions.len()];

        for face in &self.graph.faces {
            let vertex_index_a =
                self.original_data.indices[face.original_face_index * 3 + 0] as usize;
            let vertex_index_b =
                self.original_data.indices[face.original_face_index * 3 + 1] as usize;
            let vertex_index_c =
                self.original_data.indices[face.original_face_index * 3 + 2] as usize;

            let position_a = self.original_data.positions[vertex_index_a];
            let position_b = self.original_data.positions[vertex_index_b];
            let position_c = self.original_data.positions[vertex_index_c];

            let uv_a = self.original_data.tex_coords[vertex_index_a];
            let uv_b = self.original_data.tex_coords[vertex_index_b];
            let uv_c = self.original_data.tex_coords[vertex_index_c];

            let edge_vectors: SMatrix<_, 3, 2> =
                SMatrix::from_columns(&[position_b - position_a, position_c - position_a]);
            let uv_vectors: SMatrix<_, 2, 2> = SMatrix::from_columns(&[uv_b - uv_a, uv_c - uv_a]);
            let tangent_bitangent =
                edge_vectors * uv_vectors.try_inverse().unwrap_or(SMatrix::identity());
            let tangent = tangent_bitangent.column(0).normalize();

            tangents[vertex_index_a] += tangent;
            tangents[vertex_index_b] += tangent;
            tangents[vertex_index_c] += tangent;
        }

        tangents
            .iter_mut()
            .for_each(|tangent| *tangent = tangent.normalize());
        self.original_tangents = tangents;
    }

    fn construct_normal_map(&mut self, args: &Args, progress: &mut ProgressBar) {
        progress.println("Subdividing...");
        for _ in 0..args.subdivision_count {
            self.graph.subdivide();
            progress.inc(1);
        }

        if args.dump_subdivided_mesh {
            progress.println("Dumping subdivided mesh...");
            self.graph.dump();
        }

        progress.println("Computing vertex normals...");
        self.compute_normals();
        progress.inc(1);

        progress.println(&format!("Rasterizing {} faces...", self.graph.faces.len()));
        let progress_start = progress.position();
        for i in 0..self.graph.faces.len() {
            self.rasterize_triangle(args, i);

            let new_progress_position = progress_start
                + i as u64 * PROGRESS_TICKS_PER_MESH_PRIMITIVE / self.graph.faces.len() as u64;
            if new_progress_position != progress.position() {
                progress.set_position(new_progress_position);
            }
        }
        progress.set_position(progress_start + PROGRESS_TICKS_PER_MESH_PRIMITIVE);
    }

    fn rasterize_triangle(&mut self, args: &Args, face_index: usize) {
        let face = self.graph.faces[face_index];
        let info = self.face_info_for(face_index);

        let map_size = self.normal_map.texture.texels.width() as f32;

        let uv_original_a = self.original_data.tex_coords[info.original_vertex_indices[0]];
        let uv_original_b = self.original_data.tex_coords[info.original_vertex_indices[1]];
        let uv_original_c = self.original_data.tex_coords[info.original_vertex_indices[2]];

        let uv_a = bary_interp(
            uv_original_a,
            uv_original_b,
            uv_original_c,
            info.split_lambdas[0],
        ) * map_size;
        let uv_b = bary_interp(
            uv_original_a,
            uv_original_b,
            uv_original_c,
            info.split_lambdas[1],
        ) * map_size;
        let uv_c = bary_interp(
            uv_original_a,
            uv_original_b,
            uv_original_c,
            info.split_lambdas[2],
        ) * map_size;

        //println!("{:?} {:?} {:?} {:?} {:?} {:?}", uv_a, uv_b, uv_c, lambda_a, lambda_b, lambda_c);

        let min_bounds = Vector2::new(
            f32::floor(f32::max(f32::min(f32::min(uv_a.x, uv_b.x), uv_c.x), 0.0)),
            f32::floor(f32::max(f32::min(f32::min(uv_a.y, uv_b.y), uv_c.y), 0.0)),
        );
        let max_bounds = Vector2::new(
            f32::ceil(f32::min(
                f32::max(f32::max(uv_a.x, uv_b.x), uv_c.x),
                map_size,
            )),
            f32::ceil(f32::min(
                f32::max(f32::max(uv_a.y, uv_b.y), uv_c.y),
                map_size,
            )),
        );

        // Early out for zero-area triangles.
        if min_bounds.y == max_bounds.y || min_bounds.x == max_bounds.x {
            return;
        }

        for y in (min_bounds.y as i32)..(max_bounds.y as i32) {
            for x in (min_bounds.x as i32)..(max_bounds.x as i32) {
                // Convert to barycentric coordinates.
                let p = Vector2::new(x as f32 + 0.5, y as f32 + 0.5);
                let barycentric_transform =
                    match Matrix2::from_columns(&[uv_a - uv_c, uv_b - uv_c]).try_inverse() {
                        Some(barycentric_transform) => barycentric_transform,
                        None => continue,
                    };
                let lambda = barycentric_transform * (p - uv_c);
                let lambda = lambda.insert_row(2, 1.0 - lambda.x - lambda.y);

                // Are we in the triangle?
                if lambda.x < 0.0 || lambda.y < 0.0 || lambda.z < 0.0 {
                    continue;
                }

                let normal_a = self.refined_normals[face[0].index()];

                let mut normal;
                match self.graph.limit_surface_derivatives(&face, lambda) {
                    Some(derivatives) => {
                        // FIXME(pcwalton): Kind of ugly...
                        normal = derivatives.normal.normalize();
                        if normal.dot(&normal_a) < 0.0 {
                            normal = -normal;
                        }
                    }
                    None => {
                        // Fall back to linearly interpolating vertex normals.
                        // TODO: We could do better here, by increasing subdivision levels or else
                        // using Stam's eigenanalysis method.
                        let normal_b = self.refined_normals[face[1].index()];
                        let normal_c = self.refined_normals[face[2].index()];
                        normal = bary_interp(normal_a, normal_b, normal_c, lambda).normalize();
                    }
                };

                let st = Vector2::new(x as u32, y as u32);
                self.emit_normal_direction(
                    &info,
                    lambda,
                    st,
                    normal.normalize(),
                    args.object_space,
                );
            }
        }
    }

    fn face_info_for(&self, face_index: usize) -> FaceInfo {
        let face = self.graph.faces[face_index];
        let original_vertex_indices = [
            self.original_data.indices[face.original_face_index * 3 + 0] as usize,
            self.original_data.indices[face.original_face_index * 3 + 1] as usize,
            self.original_data.indices[face.original_face_index * 3 + 2] as usize,
        ];

        FaceInfo {
            original_vertex_indices,
            original_normals: [
                self.original_data.normals[original_vertex_indices[0]],
                self.original_data.normals[original_vertex_indices[1]],
                self.original_data.normals[original_vertex_indices[2]],
            ],
            original_tangents: [
                self.original_tangents[original_vertex_indices[0]],
                self.original_tangents[original_vertex_indices[1]],
                self.original_tangents[original_vertex_indices[2]],
            ],
            split_lambdas: [
                self.bary_for_split_vertex(face[0], face.original_face_index),
                self.bary_for_split_vertex(face[1], face.original_face_index),
                self.bary_for_split_vertex(face[2], face.original_face_index),
            ],
        }
    }

    fn emit_normal_direction(
        &mut self,
        info: &FaceInfo,
        lambda: Vector3<f32>,
        st: Vector2<u32>,
        normal: Vector3<f32>,
        object_space: bool,
    ) {
        //println!("{:?}", normal);
        let original_lambda = bary_interp(
            info.split_lambdas[0],
            info.split_lambdas[1],
            info.split_lambdas[2],
            lambda,
        );
        let original_normal = bary_interp(
            info.original_normals[0],
            info.original_normals[1],
            info.original_normals[2],
            original_lambda,
        )
        .normalize();
        let original_tangent = bary_interp(
            info.original_tangents[0],
            info.original_tangents[1],
            info.original_tangents[2],
            original_lambda,
        )
        .normalize();

        let normal = if object_space {
            Vector3::new(normal.x, -normal.z, normal.y)
        } else {
            // Gram-Schmidt normalization
            let tangent =
                original_tangent - original_normal.dot(&original_tangent) * original_normal;
            let bitangent = original_normal.cross(&tangent).normalize();

            Vector3::new(
                tangent.dot(&normal),
                bitangent.dot(&normal),
                original_normal.dot(&normal),
            )
        };

        self.normal_map.texture.put_unit_vector(st, normal);
    }

    fn bary_for_split_vertex(
        &self,
        vertex_index: NodeIndex,
        original_face_index: usize,
    ) -> Vector3<f32> {
        let vertex = self.graph.graph[vertex_index];
        match vertex.provenance {
            VertexProvenance::Original(original_vertex_index) => {
                let duplicates = self
                    .duplicates
                    .get(&NodeIndex::new(original_vertex_index as usize));

                let mut lambda = Vector3::zeros();
                for i in 0..3 {
                    let this_vertex_index = self.original_data.indices[original_face_index * 3 + i];
                    match duplicates {
                        None if this_vertex_index == original_vertex_index => {
                            lambda[i] = 1.0;
                            break;
                        }
                        Some(duplicates)
                            if duplicates
                                .iter()
                                .any(|&node| node.index() == this_vertex_index as usize) =>
                        {
                            lambda[i] = 1.0;
                            break;
                        }
                        _ => {}
                    }
                }
                assert_ne!(lambda, Vector3::zeros());
                return lambda;
            }
            VertexProvenance::SplitEdge { from, to } => self
                .bary_for_split_vertex(from, original_face_index)
                .lerp(&self.bary_for_split_vertex(to, original_face_index), 0.5),
        }
    }

    fn compute_normals(&mut self) {
        let mut normals = vec![Vector3::zeros(); self.graph.graph.node_count()];
        for face in &self.graph.faces {
            let a = self.graph.graph[face[0]].position;
            let b = self.graph.graph[face[1]].position;
            let c = self.graph.graph[face[2]].position;

            let mut normal = (b - a).cross(&(c - a)).normalize();

            let original_vertex = self.original_data.indices[face.original_face_index * 3];
            if normal.dot(&self.original_data.normals[original_vertex as usize]) < 0.0 {
                normal = -normal;
            }

            for &node in &face.nodes {
                normals[node.index()] += normal;
            }
        }

        normals
            .iter_mut()
            .for_each(|normal| *normal = normal.normalize());
        self.refined_normals = normals;
    }
}

fn note_primitive_skipped(
    progress: &mut ProgressBar,
    mesh: &Mesh,
    primitive: &Primitive,
    why: &str,
) {
    progress.println(&format!(
        "note: primitive {:?} skipped for mesh {:?} because {}",
        primitive.index(),
        mesh.name(),
        why
    ));
}

fn main() {
    let args = Args::parse();

    let (gltf, buffers, _) = gltf::import(&args.input).unwrap();

    // Determine progress length.
    let mut progress_length = 0;
    for mesh in gltf.meshes() {
        for _ in mesh.primitives() {
            progress_length +=
                1 + args.subdivision_count as u64 + 1 + PROGRESS_TICKS_PER_MESH_PRIMITIVE;
        }
    }
    progress_length += args.margin as u64 + 1;

    let mut progress = ProgressBar::new(progress_length);
    progress.set_style(ProgressStyle::default_bar().template("{wide_bar} {percent}%"));

    let mut normal_maps = FxHashMap::default();

    for (mesh_index, mesh) in gltf.meshes().enumerate() {
        let mesh_name = match mesh.name() {
            Some(name) => name.to_owned(),
            None => format!("{}", mesh_index),
        };

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            progress.println(&format!(
                "Reading primitive {} for mesh {:?}...",
                primitive_index, mesh_name
            ));

            let mut normal_map = match primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
            {
                None => {
                    note_primitive_skipped(
                        &mut progress,
                        &mesh,
                        &primitive,
                        "its material doesn't have a base color texture",
                    );
                    continue;
                }
                Some(texture_info) => {
                    let texture = texture_info.texture();
                    normal_maps.entry(texture.index()).or_insert_with(|| {
                        let name = match texture.name() {
                            Some(name) => name.to_owned(),
                            None => format!("{}", texture.index()),
                        };
                        NormalMap::new(name, args.size)
                    })
                }
            };

            let reader =
                primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &*data.0));
            if primitive.mode() != Mode::Triangles {
                note_primitive_skipped(
                    &mut progress,
                    &mesh,
                    &primitive,
                    &format!("it's composed of {:?}, not triangles", primitive.mode()),
                );
                continue;
            }
            let positions = match reader.read_positions() {
                Some(positions) => positions,
                None => {
                    note_primitive_skipped(
                        &mut progress,
                        &mesh,
                        &primitive,
                        "it's missing positions",
                    );
                    continue;
                }
            };
            let normals = match reader.read_normals() {
                Some(normals) => normals,
                None => {
                    note_primitive_skipped(
                        &mut progress,
                        &mesh,
                        &primitive,
                        "it's missing normals",
                    );
                    continue;
                }
            };
            let tex_coords = match reader.read_tex_coords(0) {
                Some(tex_coords) => tex_coords,
                None => {
                    note_primitive_skipped(
                        &mut progress,
                        &mesh,
                        &primitive,
                        "it's missing texture coordinates (UVs) 0",
                    );
                    continue;
                }
            };
            let indices = match reader.read_indices() {
                Some(indices) => indices,
                None => {
                    note_primitive_skipped(
                        &mut progress,
                        &mesh,
                        &primitive,
                        "it's missing indices",
                    );
                    continue;
                }
            };

            let mut processor = PrimitiveProcessor::new(
                &mut normal_map,
                positions,
                normals,
                tex_coords.into_f32(),
                indices.into_u32(),
            );
            processor.compute_original_tangents();
            processor.merge_duplicates();

            progress.inc(1);
            processor.construct_normal_map(&args, &mut progress);
        }
    }

    progress.println("Creating margins...");
    for _ in 0..args.margin {
        for normal_map in normal_maps.values_mut() {
            normal_map.texture.inflate_margin();
            progress.inc(1);
        }
    }

    progress.println("Writing normal maps...");
    for normal_map in normal_maps.values() {
        normal_map.texture.write_to(&PathBuf::from(format!(
            "normal-map-{}.png",
            normal_map.name
        )));
    }
    progress.finish();
}
