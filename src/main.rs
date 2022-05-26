// loop-baker/src/main.rs

#![feature(let_else)]

use clap::Parser;
use derive_more::Index;
use fxhash::FxHashMap;
use gltf::accessor::Iter;
use gltf::mesh::util::indices::{CastingIter as IndicesCastingIter, U32};
use gltf::mesh::util::tex_coords::{CastingIter as TexCoordsCastingIter, F32};
use gltf::mesh::Mode;
use image::{ImageFormat, Rgb, RgbImage};
use nalgebra::{Matrix2, SMatrix, SVector, Vector2, Vector3};
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use smallvec::SmallVec;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};

const NORMAL_MAP_SIZE: u32 = 1024;

const MAX_SUBDIVISIONS: u32 = 3;

#[derive(Parser)]
struct Args {
    input: PathBuf,
}

struct NormalMap {
    texels: RgbImage,
}

impl NormalMap {
    fn new() -> NormalMap {
        NormalMap {
            texels: RgbImage::new(NORMAL_MAP_SIZE, NORMAL_MAP_SIZE),
        }
    }

    fn put_pixel(&mut self, pos: Vector2<u32>, color: [u8; 3]) {
        self.texels.put_pixel(pos.x, pos.y, Rgb(color))
    }

    fn write_to(&self, path: &Path) {
        self.texels
            .save_with_format(path, ImageFormat::Png)
            .unwrap();
    }
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

fn bary_interp<T>(a: T, b: T, c: T, lambda: Vector3<f32>) -> T
where
    T: Add<Output = T> + Mul<f32, Output = T>,
{
    a * lambda.x + b * lambda.y + c * lambda.z
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

    fn limit_surface_normal(&self, face: &Face, lambda: Vector3<f32>) -> Option<Vector3<f32>> {
        #[rustfmt::skip]
        static PHI: Lazy<SMatrix<f32, 12, 15>> = Lazy::new(|| {
            let phi: SMatrix<f32, 12, 15> = SMatrix::from_row_slice(&[
                6.,  0.,  0., -12., -12., -12.,  8.,  12.,  12.,  8., -1., -2.,  0., -2., -1.,
                1.,  4.,  2.,   6.,   6.,   0., -4.,  -6., -12., -4., -1., -2.,  0.,  4.,  2.,
                1.,  2.,  4.,   0.,   6.,   6., -4., -12.,  -6., -4.,  2.,  4.,  0., -2., -1.,
                0.,  0.,  0.,   0.,   0.,   0.,  2.,   6.,   6.,  2., -1., -2.,  0., -2., -1.,
                0.,  0.,  0.,   0.,   0.,   0.,  0.,   0.,   0.,  0.,  0.,  0.,  0.,  2.,  1.,
                0.,  0.,  0.,   0.,   0.,   0.,  0.,   0.,   0.,  2.,  0.,  0.,  0., -2., -1.,
                1., -2.,  2.,   0.,  -6.,   0.,  2.,   6.,   0., -4., -1., -2.,  0.,  4.,  2.,
                1., -4., -2.,   6.,   6.,   0., -4.,  -6.,   0.,  2.,  1.,  2.,  0., -2., -1.,
                1., -2., -4.,   0.,   6.,   6.,  2.,   0.,  -6., -4., -1., -2.,  0.,  2.,  1.,
                1.,  2., -2.,   0.,  -6.,   0., -4.,   0.,   6.,  2.,  2.,  4.,  0., -2., -1.,
                0.,  0.,  0.,   0.,   0.,   0.,  2.,   0.,   0.,  0., -1., -2.,  0.,  0.,  0.,
                0.,  0.,  0.,   0.,   0.,   0.,  0.,   0.,   0.,  0.,  1.,  2.,  0.,  0.,  0.,
            ]);
            phi.scale(1.0 / 12.0)
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

        let v10 = self.node_across_from(v1, v2, v3).unwrap();
        let v4 = self.node_across_from(v2, v3, v1).unwrap();
        let v7 = self.node_across_from(v3, v1, v2).unwrap();

        let v9 = self.node_across_from(v1, v10, v2).unwrap();
        let v8 = self.node_across_from(v1, v7, v3).unwrap();
        let v11 = self.node_across_from(v2, v10, v1).unwrap();
        let v12 = self.node_across_from(v2, v4, v3).unwrap();
        let v6 = self.node_across_from(v3, v7, v1).unwrap();
        let v5 = self.node_across_from(v3, v4, v2).unwrap();

        let (v, w) = (lambda.y, lambda.z);
        let dv: SVector<f32, 15> = SVector::from_column_slice(&[
            0.0,
            1.0,
            0.0,
            2.0 * v,
            w,
            0.0,
            3.0 * v * v,
            2.0 * v * w,
            w * w,
            0.0,
            4.0 * v * v * v,
            3.0 * v * v * w,
            2.0 * v * w * w,
            w * w * w,
            0.0,
        ]);
        let dw: SVector<f32, 15> = SVector::from_column_slice(&[
            0.0,
            0.0,
            1.0,
            0.0,
            v,
            2.0 * w,
            0.0,
            v * w,
            2.0 * v * w,
            3.0 * w * w * w,
            0.0,
            v * v * v,
            2.0 * v * v * w,
            3.0 * v * w * w,
            4.0 * w * w * w,
        ]);

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

        let weights = x * *PHI;
        let tangent_v = weights * dv;
        let tangent_w = weights * dw;
        Some(tangent_v.cross(&tangent_w).normalize())
    }

    fn node_across_from(
        &self,
        edge_v0: NodeIndex,
        edge_v1: NodeIndex,
        across_from: NodeIndex,
    ) -> Option<NodeIndex> {
        self.graph.neighbors(edge_v0).find(|v| {
            self.graph.neighbors(edge_v1).any(|node| node == *v) && *v != across_from
        })
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

    fn update_after_subdivision(&self, face: &mut Face, lambda: &mut Vector3<f32>) -> bool {
        //  v=1 o
        //      |\
        //      +-+
        //      | |\
        //      o-+-o
        //  u,v=0   u=1

        let (p_u, p_v, p_w) = (face[0], face[1], face[2]);

        let mut midpoints = [NodeIndex::new(0); 3];
        for i in 0..3 {
            let (a, b) = (face[i], face[(i + 1) % 3]);
            // TODO: maybe slow?
            midpoints[i] = match self
                .graph
                .neighbors(a)
                .find(|&node| self.graph.neighbors(node).any(|neighbor| neighbor == b))
            {
                None => return false,
                Some(midpoint) => midpoint,
            }
        }
        let (p_uv, p_vw, p_wu) = (midpoints[0], midpoints[1], midpoints[2]);

        let (u, v) = (lambda[0], lambda[1]);
        let (new_u, new_v) = if v >= 0.5 {
            // top left triangle
            *face = Face {
                nodes: [p_uv, p_v, p_vw],
                original_face_index: face.original_face_index,
            };
            (u * 2.0, (v - 0.5) * 2.0)
        } else if u >= 0.5 {
            // bottom right triangle
            *face = Face {
                nodes: [p_u, p_uv, p_wu],
                original_face_index: face.original_face_index,
            };
            ((u - 0.5) * 2.0, v * 2.0)
        } else if v < 0.5 - u {
            // bottom left triangle: v >= -u + 1/2
            *face = Face {
                nodes: [p_wu, p_w, p_vw],
                original_face_index: face.original_face_index,
            };
            (u * 2.0, v * 2.0)
        } else {
            // middle triangle
            *face = Face {
                nodes: [p_vw, p_wu, p_uv],
                original_face_index: face.original_face_index,
            };
            (1.0 - u * 2.0, 1.0 - v * 2.0)
        };
        *lambda = Vector3::new(new_u, new_v, 1.0 - new_u - new_v);

        true
    }

    fn compute_normals(&self) -> Vec<Vector3<f32>> {
        todo!()
    }
}

struct PrimitiveProcessor<'a> {
    normal_map: &'a mut NormalMap,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tex_coords: Vec<[f32; 2]>,
    indices: Vec<u32>,
    graph: MeshGraph,
    duplicates: FxHashMap<NodeIndex, SmallVec<[NodeIndex; 2]>>,
}

impl<'a> PrimitiveProcessor<'a> {
    fn new(
        normal_map: &'a mut NormalMap,
        positions: Iter<[f32; 3]>,
        normals: Iter<[f32; 3]>,
        tex_coords: TexCoordsCastingIter<F32>,
        indices: IndicesCastingIter<U32>,
    ) -> Self {
        let positions: Vec<_> = positions.collect();
        let normals = normals.collect();
        let tex_coords = tex_coords.collect();
        let indices: Vec<_> = indices.collect();

        // Build a graph.
        let mut graph = UnGraph::new_undirected();
        let mut faces = Vec::with_capacity(indices.len() / 3);
        for i in 0..positions.len() {
            let node_index = graph.add_node(Vertex {
                provenance: VertexProvenance::Original(i as u32),
                position: Vector3::from_column_slice(&positions[i]),
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
            positions,
            normals,
            tex_coords,
            indices,
            graph: MeshGraph::new(graph, faces),
            duplicates: FxHashMap::default(),
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

    fn process(&mut self) {
        eprintln!("Subdividing...");
        for _ in 0..MAX_SUBDIVISIONS {
            self.graph.subdivide();
        }

        //eprintln!("Computing vertex normals...");
        //self.subdivided_normals = self.graph.compute_normals();

        eprintln!("Rasterizing...");
        for i in 0..self.graph.faces.len() {
            let mut rng = StdRng::seed_from_u64(i as u64);
            let color = [rng.gen(), rng.gen(), rng.gen()];
            self.rasterize_triangle(i, color)
        }
    }

    fn rasterize_triangle(&mut self, face_index: usize, color: [u8; 3]) {
        let face = &self.graph.faces[face_index];

        let original_vertex_index_a = self.indices[face.original_face_index * 3 + 0] as usize;
        let original_vertex_index_b = self.indices[face.original_face_index * 3 + 1] as usize;
        let original_vertex_index_c = self.indices[face.original_face_index * 3 + 2] as usize;

        let lambda_a = self.bary_for_split_vertex(face[0], face.original_face_index);
        let lambda_b = self.bary_for_split_vertex(face[1], face.original_face_index);
        let lambda_c = self.bary_for_split_vertex(face[2], face.original_face_index);

        let map_size = NORMAL_MAP_SIZE as f32;

        let uv_original_a = Vector2::from(self.tex_coords[original_vertex_index_a]);
        let uv_original_b = Vector2::from(self.tex_coords[original_vertex_index_b]);
        let uv_original_c = Vector2::from(self.tex_coords[original_vertex_index_c]);

        let uv_a = bary_interp(uv_original_a, uv_original_b, uv_original_c, lambda_a) * map_size;
        let uv_b = bary_interp(uv_original_a, uv_original_b, uv_original_c, lambda_b) * map_size;
        let uv_c = bary_interp(uv_original_a, uv_original_b, uv_original_c, lambda_c) * map_size;

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

        for y in (min_bounds.y as i32)..(max_bounds.y as i32) {
            'next: for x in (min_bounds.x as i32)..(max_bounds.x as i32) {
                // Convert to barycentric coordinates.
                let p = Vector2::new(x as f32 + 0.5, y as f32 + 0.5);
                let Some(barycentric_transform) =
                    Matrix2::from_columns(&[uv_a - uv_c, uv_b - uv_c]).try_inverse() else {
                        continue
                    };
                let lambda = barycentric_transform * (p - uv_c);
                let lambda = lambda.insert_row(2, 1.0 - lambda.x - lambda.y);

                // Are we in the triangle?
                if lambda.x < 0.0 || lambda.y < 0.0 || lambda.z < 0.0 {
                    continue;
                }

                let normal = match self.graph.limit_surface_normal(face, lambda) {
                    Some(normal) => normal,
                    None => {
                        /*
                        let (mut face, mut lambda) = (*face, lambda);
                        let mut graph = self.graph.clone();
                        for _ in 0..MAX_SUBDIVISIONS {
                            graph.subdivide();
                            if !graph.update_after_subdivision(&mut face, &mut lambda) {
                                continue 'next;
                            }

                            if graph.face_is_regular(&face) {
                                break;
                            }
                        }

                        normal = graph.limit_surface_normal(&face, lambda);
                        //println!("{},{}", x, y);
                        */
                        Vector3::new(0.0, 0.0, -1.0)
                    }
                };

                //println!("{:?}", normal);

                // FIXME(pcwalton): This needs to be relative to the interpolated surface normal!!!
                let st = Vector2::new(x as u32, y as u32);
                self.normal_map.put_pixel(
                    st,
                    [
                        ((normal.x + 1.0) * 0.5 * 255.0).round() as u8,
                        ((normal.y + 1.0) * 0.5 * 255.0).round() as u8,
                        ((-normal.z + 1.0) * 0.5 * 255.0).round() as u8,
                    ],
                );
            }
        }
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
                    let this_vertex_index = self.indices[original_face_index * 3 + i];
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
}

fn main() {
    let args = Args::parse();

    let (gltf, buffers, _) = gltf::import(&args.input).unwrap();
    for (mesh_index, mesh) in gltf.meshes().enumerate() {
        let mesh_name = match mesh.name() {
            Some(name) => name.to_owned(),
            None => format!("{}", mesh_index),
        };

        let mut normal_map = NormalMap::new();

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            let reader =
                primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &*data.0));
            if primitive.mode() != Mode::Triangles {
                eprintln!(
                    "note: primitive {:?} skipped for mesh {:?} because it's rendering {:?}, \
                         not triangles",
                    primitive_index,
                    mesh.name(),
                    primitive.mode()
                );
                continue;
            }
            let Some(positions) = reader.read_positions() else {
                    eprintln!(
                        "note: primitive {:?} skipped for mesh {:?} because it's missing positions",
                        primitive_index,
                        mesh_name);
                    continue
                };
            let Some(normals) = reader.read_normals() else {
                    eprintln!(
                        "note: primitive {:?} skipped for mesh {:?} because it's missing normals",
                        primitive_index,
                        mesh_name);
                    continue
                };
            let Some(tex_coords) = reader.read_tex_coords(0) else {
                    eprintln!(
                        "note: primitive {:?} skipped for mesh {:?} because it's missing texture \
                         coordinates (UVs) 0",
                        primitive_index,
                        mesh_name);
                    continue
                };
            let Some(indices) = reader.read_indices() else {
                    eprintln!(
                        "note: primitive {:?} skipped for mesh {:?} because it's missing indices",
                        primitive_index,
                        mesh_name);
                    continue
                };

            let mut processor = PrimitiveProcessor::new(
                &mut normal_map,
                positions,
                normals,
                tex_coords.into_f32(),
                indices.into_u32(),
            );
            processor.merge_duplicates();
            processor.process();
        }

        normal_map.write_to(&PathBuf::from(format!("normal-map-{}.png", mesh_name)));
    }
}
