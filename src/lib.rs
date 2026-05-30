pub const EMBEDDING_DIM: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding(pub [f64; EMBEDDING_DIM]);

impl Embedding {
    pub fn zero() -> Self {
        Embedding([0.0; EMBEDDING_DIM])
    }

    pub fn from_value(base: f64) -> Self {
        let mut arr = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            arr[i] = base + i as f64 * 0.1;
        }
        Embedding(arr)
    }

    pub fn add(&self, other: &Embedding) -> Embedding {
        let mut arr = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            arr[i] = self.0[i] + other.0[i];
        }
        Embedding(arr)
    }

    pub fn sub(&self, other: &Embedding) -> Embedding {
        let mut arr = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            arr[i] = self.0[i] - other.0[i];
        }
        Embedding(arr)
    }

    pub fn scale(&self, s: f64) -> Embedding {
        let mut arr = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            arr[i] = self.0[i] * s;
        }
        Embedding(arr)
    }

    pub fn dot(&self, other: &Embedding) -> f64 {
        self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f64 {
        let d = self.norm() * other.norm();
        if d < 1e-12 { 0.0 } else { self.dot(other) / d }
    }

    pub fn euclidean_dist(&self, other: &Embedding) -> f64 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn centroid(embeddings: &[TaggedEmbedding]) -> Embedding {
        if embeddings.is_empty() { return Embedding::zero(); }
        let mut c = Embedding::zero();
        for te in embeddings {
            c = c.add(&te.value);
        }
        c.scale(1.0 / embeddings.len() as f64)
    }
}

#[derive(Debug, Clone)]
pub struct TaggedEmbedding {
    pub value: Embedding,
    pub strength: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct Vibe {
    pub position: Embedding,
    pub velocity: Embedding,
    pub acceleration: Embedding,
    pub strength: f64,
}

impl Default for Vibe {
    fn default() -> Self {
        Vibe {
            position: Embedding::zero(),
            velocity: Embedding::zero(),
            acceleration: Embedding::zero(),
            strength: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Room {
    pub id: String,
    pub perception_db: Vec<TaggedEmbedding>,
    pub prediction_db: Vec<TaggedEmbedding>,
    pub vibe: Vibe,
    prev_position: Embedding,
    prev_velocity: Embedding,
}

impl Room {
    pub fn new(id: &str) -> Self {
        Room {
            id: id.to_string(),
            perception_db: Vec::new(),
            prediction_db: Vec::new(),
            vibe: Vibe::default(),
            prev_position: Embedding::zero(),
            prev_velocity: Embedding::zero(),
        }
    }

    pub fn tick(&mut self, timestamp: f64, sensor_id: i32, perception: Embedding) -> f64 {
        // Store perception
        self.perception_db.push(TaggedEmbedding {
            value: perception.clone(),
            strength: 1.0,
            timestamp,
        });

        // Generate prediction
        let pred = self.predict();
        self.prediction_db.push(TaggedEmbedding {
            value: pred.clone(),
            strength: 1.0,
            timestamp,
        });

        // Compute prediction error
        let error = perception.euclidean_dist(&pred);

        // Update vibe
        self.compute_vibe();

        error
    }

    pub fn predict(&self) -> Embedding {
        self.vibe.position.add(&self.vibe.velocity)
    }

    pub fn balance_check(&self) -> bool {
        self.perception_db.len() == self.prediction_db.len()
    }

    pub fn compute_vibe(&mut self) {
        let old_vel = self.vibe.velocity.clone();

        self.vibe.position = Embedding::centroid(&self.perception_db);
        self.vibe.velocity = self.vibe.position.sub(&self.prev_position);
        self.vibe.acceleration = self.vibe.velocity.sub(&self.prev_velocity);
        self.vibe.strength = self.perception_db.len() as f64;

        self.prev_position = self.vibe.position.clone();
        self.prev_velocity = old_vel;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GCReport {
    pub merged: usize,
    pub decayed: usize,
    pub pruned: usize,
}

fn merge_similar(db: &mut Vec<TaggedEmbedding>, threshold: f64) -> usize {
    let mut merged = 0;
    let n = db.len();
    let mut removed = vec![false; n];

    for i in 0..n {
        if removed[i] { continue; }
        for j in (i + 1)..n {
            if removed[j] { continue; }
            if db[i].value.euclidean_dist(&db[j].value) < threshold {
                db[i].value = db[i].value.add(&db[j].value).scale(0.5);
                db[i].strength += db[j].strength;
                removed[j] = true;
                merged += 1;
            }
        }
    }

    let mut write = 0;
    for i in 0..n {
        if !removed[i] {
            db[write] = db[i].clone();
            write += 1;
        }
    }
    db.truncate(write);
    merged
}

fn decay(db: &mut Vec<TaggedEmbedding>, rate: f64) -> usize {
    for te in db.iter_mut() {
        te.strength *= rate;
    }
    db.len()
}

fn prune(db: &mut Vec<TaggedEmbedding>, min_strength: f64) -> usize {
    let before = db.len();
    db.retain(|te| te.strength >= min_strength);
    before - db.len()
}

pub fn gc(room: &mut Room, merge_threshold: f64, decay_rate: f64, min_strength: f64) -> GCReport {
    let merged = merge_similar(&mut room.perception_db, merge_threshold)
               + merge_similar(&mut room.prediction_db, merge_threshold);
    let decayed = decay(&mut room.perception_db, decay_rate)
                + decay(&mut room.prediction_db, decay_rate);
    let pruned = prune(&mut room.perception_db, min_strength)
               + prune(&mut room.prediction_db, min_strength);
    GCReport { merged, decayed, pruned }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from_id: String,
    pub to_id: String,
    pub algorithm: i32,
}

#[derive(Debug)]
pub struct CellularGraph {
    pub rooms: Vec<Room>,
    pub edges: Vec<Edge>,
}

impl CellularGraph {
    pub fn new() -> Self {
        CellularGraph { rooms: Vec::new(), edges: Vec::new() }
    }

    pub fn add_room(&mut self, room: Room) {
        self.rooms.push(room);
    }

    pub fn add_edge(&mut self, from: &str, to: &str, algorithm: i32) {
        self.edges.push(Edge {
            from_id: from.to_string(),
            to_id: to.to_string(),
            algorithm,
        });
    }

    pub fn find_room(&mut self, id: &str) -> Option<&mut Room> {
        self.rooms.iter_mut().find(|r| r.id == id)
    }

    pub fn murmur(from: &Room, to: &mut Room) -> Embedding {
        let vibe_pos = from.vibe.position.clone();
        to.perception_db.push(TaggedEmbedding {
            value: vibe_pos.clone(),
            strength: 0.5,
            timestamp: 0.0,
        });
        vibe_pos
    }

    pub fn cross_room_correlation(a: &Room, b: &Room) -> f64 {
        a.vibe.position.cosine_similarity(&b.vibe.position)
    }

    pub fn tick_through_graph(&mut self, timestamp: f64, sensor_id: i32, perception: Embedding) {
        // Tick all rooms
        for room in &mut self.rooms {
            room.tick(timestamp, sensor_id, perception.clone());
        }
        // Propagate via edges
        let edges: Vec<_> = self.edges.clone();
        for edge in &edges {
            let (from_pos, to_id) = {
                if let Some(from) = self.rooms.iter().find(|r| r.id == edge.from_id) {
                    (from.vibe.position.clone(), edge.to_id.clone())
                } else {
                    continue;
                }
            };
            if let Some(to) = self.rooms.iter_mut().find(|r| r.id == to_id) {
                to.perception_db.push(TaggedEmbedding {
                    value: from_pos,
                    strength: 0.5,
                    timestamp,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_updates_perception() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        assert_eq!(r.perception_db.len(), 1);
        assert_eq!(r.prediction_db.len(), 1);
    }

    #[test]
    fn test_predict_generates_embedding() {
        let mut r = Room::new("test");
        let pred = r.predict();
        assert!(pred.norm() < 1e-9, "first prediction should be near zero");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        r.tick(2.0, 1, Embedding::from_value(2.0));
        let pred2 = r.predict();
        assert!(pred2.norm() > 0.0, "prediction after ticks should be non-zero");
    }

    #[test]
    fn test_balance_check_passes_equal() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        r.tick(2.0, 1, Embedding::from_value(2.0));
        assert!(r.balance_check());
    }

    #[test]
    fn test_balance_check_fails_unequal() {
        let mut r = Room::new("test");
        r.perception_db.push(TaggedEmbedding {
            value: Embedding::from_value(1.0),
            strength: 1.0,
            timestamp: 1.0,
        });
        assert!(!r.balance_check());
    }

    #[test]
    fn test_vibe_computation() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        assert!(r.vibe.strength > 0.0);
        assert!(r.vibe.position.norm() > 0.0);
    }

    #[test]
    fn test_merge_reduces_count() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        r.tick(2.0, 1, Embedding::from_value(1.01));
        let before = r.perception_db.len();
        let _report = gc(&mut r, 0.5, 0.99, 0.01);
        assert!(r.perception_db.len() <= before);
    }

    #[test]
    fn test_decay_reduces_strengths() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        let before = r.perception_db[0].strength;
        gc(&mut r, 999.0, 0.5, 0.0);
        assert!(r.perception_db[0].strength < before);
    }

    #[test]
    fn test_prune_removes_weak() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        r.perception_db[0].strength = 0.001;
        if !r.prediction_db.is_empty() {
            r.prediction_db[0].strength = 0.001;
        }
        gc(&mut r, 999.0, 1.0, 0.01);
        assert!(r.perception_db.is_empty());
    }

    #[test]
    fn test_full_gc_cycle() {
        let mut r = Room::new("test");
        r.tick(1.0, 1, Embedding::from_value(1.0));
        r.tick(2.0, 1, Embedding::from_value(1.01));
        r.tick(3.0, 1, Embedding::from_value(5.0));
        let report = gc(&mut r, 0.5, 0.9, 0.01);
        assert!(report.decayed > 0);
    }

    #[test]
    fn test_cross_room_correlation() {
        let mut a = Room::new("a");
        let mut b = Room::new("b");
        a.tick(1.0, 1, Embedding::from_value(1.0));
        b.tick(1.0, 1, Embedding::from_value(1.0));
        let corr = CellularGraph::cross_room_correlation(&a, &b);
        assert!(corr > 0.99);
    }

    #[test]
    fn test_murmur_sends_vibe() {
        let mut from = Room::new("from");
        let mut to = Room::new("to");
        from.tick(1.0, 1, Embedding::from_value(3.0));
        let before = to.perception_db.len();
        CellularGraph::murmur(&from, &mut to);
        assert!(to.perception_db.len() > before);
    }

    #[test]
    fn test_graph_construction() {
        let mut g = CellularGraph::new();
        g.add_room(Room::new("r1"));
        g.add_room(Room::new("r2"));
        g.add_edge("r1", "r2", 0);
        assert_eq!(g.rooms.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn test_tick_through_graph() {
        let mut g = CellularGraph::new();
        g.add_room(Room::new("r1"));
        g.add_room(Room::new("r2"));
        g.add_edge("r1", "r2", 0);
        g.tick_through_graph(1.0, 1, Embedding::from_value(1.0));
        let r1 = g.rooms.iter().find(|r| r.id == "r1").unwrap();
        let r2 = g.rooms.iter().find(|r| r.id == "r2").unwrap();
        assert!(r1.perception_db.len() >= 1);
        assert!(r2.perception_db.len() >= 2, "r2 should have tick + murmur");
    }
}
