pub fn perfect_hash(select_from: &[(u32, u32)]) -> u32 {
    let mut place = 1;
    let mut total = 0;
    for select_from in select_from {
        assert!(select_from.0 < select_from.1);
        assert!(select_from.1 != 0);
        total += select_from.0 * place;
        place *= select_from.1;
    }
    total
}


pub struct Lookup {
    table: Vec<f64>,
    len: usize,
}

impl Lookup {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut scratch = [0u8; 8];
        let mut len = 0;
        let mut table = Vec::with_capacity(bytes.len() / 8);
        for i in 0..bytes.len() / 8 {
            scratch.copy_from_slice(&bytes[i*8..i*8+8]);
            let f = f64::from_le_bytes(scratch);
            if f != 0.0 {
                len += 1;
            }
            table.push(f);
        }
        Lookup { table, len }
    }
    pub fn lookup(&self, index: u32) -> Option<f64> {
        let index = index as usize;
        if self.table.len() <= index {
            return None;
        }
        let result = self.table[index];
        if result == 0.0 {
            return None;
        }
        Some(result)
    }
    #[allow(unused)]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.table.len());
        for elem in &self.table {
            result.extend(elem.to_le_bytes());
        }
        result
    }
    pub fn insert(&mut self, index: u32, value: f64) {
        assert!(value != 0.0);
        let index = index as usize;
        while self.table.len() <= index {
            self.table.push(0.0);
        }
        self.len += 1;
        if self.len % 1_000_000 == 0 {
            let percent = ((self.len as f64 / self.table.len() as f64) * 10000.0).round() / 100.0;
            println!("{}%", percent);
        }
        debug_assert!(self.table[index] == 0.0);
        self.table[index] = value;
    }    
}