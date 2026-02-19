use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::mem::transmute;
use std::time::Instant;
struct F32 {
    inner: AtomicU32
}
impl F32 {
    #[inline]
    fn get(&self) -> f32 {
        unsafe { transmute(self.inner.load(SeqCst)) }
    }
    fn set(&self, value: f32) {
        self.inner.store(unsafe { transmute(value) }, SeqCst);
    }
    fn new(value: f32) -> Self {
        Self {
            inner: AtomicU32::new(unsafe { transmute(value) })
        }
    }
}

pub struct PerfectHasher {
    place: u32,
    total: u32,
}

impl PerfectHasher {
    pub fn new() -> Self {
        Self {
            place: 1,
            total: 0,
        }
    }
    pub fn update(&mut self, value: u32, max: u32) {
        assert!(value < max);
        assert!(max != 0);
        self.total += value * self.place;
        self.place *= max;

    }
    pub fn digest(self) -> u32 {
        self.total
    }
}


pub trait Lookup {
    type Key;
    type Value;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn insert(&mut self, key: Self::Key, value: Self::Value);
    fn get(&self, key: &Self::Key) -> Option<Self::Value>;
}

// TODO: SparseLookup

pub struct DenseLookup {
    table: Vec<F32>,
    len: usize,
    start: Instant,
}

impl Lookup for DenseLookup {
    type Key = u32;
    type Value = f32;
    fn len(&self) -> usize {
        self.len
    }
    fn capacity(&self) -> usize {
        self.table.capacity()
    }
    fn get(&self, key: &u32) -> Option<Self::Value> {
        let index = *key as usize;
        if self.table.len() <= index {
            return None;
        }
        let result = self.table[index].get();
        if result == 0.0 {
            return None;
        }
        Some(result)
    }
    fn insert(&mut self, key: u32, value: f32) {
        assert!(value != 0.0);
        let index = key as usize;
        self.len += 1;
        if self.len % 1_000_000 == 0 {
            let percent = ((self.len as f32 / self.table.len() as f32) * 10000.0).round() / 100.0;
            println!("{}%", percent);
        }
        debug_assert!(self.table[index].get() == 0.0);
        self.table[index].set(value);
    }   
}

impl DenseLookup {
    pub fn new(capacity: usize) -> Self {
        let mut table = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            table.push(F32::new(0.0f32));
        }
        DenseLookup { table, len: 0, start: Instant::now() }
    }
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut scratch = [0u8; 4];
        let mut len = 0;
        let mut table = Vec::with_capacity(bytes.len() / 4);
        for i in 0..bytes.len() / 8 {
            scratch.copy_from_slice(&bytes[i*8..i*8+8]);
            let f = f32::from_le_bytes(scratch);
            if f != 0.0 {
                len += 1;
            }
            table.push(F32::new(f));
        }
        DenseLookup { table, len, start: Instant::now() }
    }
    
    #[allow(unused)]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.table.len() * 4);
        for elem in &self.table {
            result.extend(elem.get().to_le_bytes());
        }
        result
    }
}