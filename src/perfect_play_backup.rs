use std::time::Instant;

use {
    std::{hash::Hash},
    rand::{Rng, thread_rng},
    arrayvec::ArrayVec,
};

type Coords = ArrayVec<(isize, isize), 4>;


#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
enum Tile {
    Filled, // Has a piece played
    Empty, // Could have piece played
    Null, // Not a part of the game
    Bonus(u32), // Has a star
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Piece {
    One,
    Two,
    ThreeRow,
    FourRow,
    ThreeL,
    FourL,
    FourT,
    FourSquare,
    FourZ,
}

fn die_roll<R: Rng>(r: &mut R) -> u32 {
    let d4 = r.gen_range(1..=4);
    let d6 = r.gen_range(1..=6);

    d4 + d6
}

fn roll_to_piece(roll: u32) -> Piece {
    use Piece::*;
    match roll {
        2 => Two,
        3 => FourRow,
        4 => FourL,
        5 => FourT,
        6 => FourSquare,
        7 => FourZ,
        8 => ThreeL,
        9 => ThreeRow,
        10 => One,
        _ => unreachable!()
    }
}

impl Piece {
    fn random_from_dice<R: Rng>(r: &mut R) -> Self {
        roll_to_piece(die_roll(r))
    }
    /// Transformations which when evaluated over each placement point yields a unique
    /// and complete set of movement options.
    /// All of them include (0, 0).
    /// Coordinates are (X, Y)
    /// If a tile is directly above another tile, it goes right after the tile it is above.
    fn transformations(&self) -> &'static [&'static [(isize, isize)]] {
        use Piece::*;
        // TODO: Write a unit test showing these are complete and unique.
        match self {
            One => &[
                &[(0, 0)]
            ],
            Two => &[
                &[(0, 0), (0, 1)],
                &[(0, 0), (1, 0)],
            ],
            FourRow => &[
                &[(0, 0), (0, 1), (0, 2), (0, 3)],
                &[(0, 0), (1, 0), (2, 0), (3, 0)],
            ],
            FourL => &[
                &[(0, 0), (0, 1), (1, 0), (2, 0)],
                &[(0, 0), (0, 1), (-1, 0), (-2, 0)],
                &[(0, 0), (0, 1), (0, 2), (-1, 2)],
                &[(0, 0), (0, 1), (0, 2), (1, 2)],
                &[(0, 0), (0, 1), (1, 1), (2, 1)],
                &[(0, 0), (0, 1), (-1, 1), (-2, 1)],
                &[(0, 0), (-1, 0), (-1, 1), (-1, 2)],
                &[(0, 0), (1, 0), (1, 1), (1, 2)],
            ],
            FourSquare => &[
                &[(0, 0), (0, 1), (1, 0), (1, 1)],
            ],
            FourT => &[
                &[(0, 0), (0, 1), (-1, 1), (1, 1)],
                &[(0, 0), (0, 1), (-1, 0), (1, 0)],
                &[(0, 0), (0, 1), (0, 2), (1, 1)],
                &[(0, 0), (0, 1), (0, 2), (-1, 1)],
            ],
            ThreeRow => &[
                &[(0, 0), (0, 1), (0, 2)],
                &[(0, 0), (1, 0), (2, 0)],
            ],
            ThreeL => &[
                &[(0, 0), (0, 1), (1, 1)],
                &[(0, 0), (0, 1), (-1, 1)],
                &[(0, 0), (1, 0), (1, 1)],
                &[(0, 0), (-1, 0), (-1, 1)],
            ],
            FourZ => &[
                &[(0, 0), (1, 0), (1, 1), (2, 1)],
                &[(0, 0), (-1, 0), (-1, 1), (-2, 1)],
                &[(0, 0), (0, 1), (-1, 1), (-1, 2)],
                &[(0, 0), (0, 1), (1, 1), (1, 2)],
            ],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Bank {
    Unused,
    Stored(Piece),
    Used,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
struct GameState {
    bank: Bank,
    piece: Option<Piece>,
    tower: Tower,
    bonus: Option<u32>,
}



fn available_moves(game: &GameState) -> Vec<Move> {
    if game.bonus.is_some() {
        return Vec::new();
    }
    let piece = game.piece.unwrap();

    let mut moves = Vec::new();
    let mut pieces = ArrayVec::<_, 2>::new();

    pieces.push((Source::Dice, piece));

    match game.bank {
        Bank::Unused => {
            moves.push(Move::BankPiece);
        },
        Bank::Stored(p) => {
            if p != piece {
                pieces.push((Source::Bank, p));
            }
        }
        Bank::Used => {},
    }

    for row in 0..7 {
        for column in 0..15 {
            if game.tower.get((row, column)) == Tile::Empty {
                let cell = (row as isize, column as isize);
                for (source, piece) in pieces.iter() {
                    for transformation in piece.transformations() {
                        let placement = transform(cell, &transformation);
                        if placement_is_legal(&game.tower, &placement) {
                            moves.push(Move::PlacePiece(PlacePiece { source: *source, placement }));
                        }
                    }
                }
                break;
            }
        }
    }
    
    moves
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Source {
    Bank,
    Dice,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Move {
    BankPiece,
    PlacePiece(PlacePiece),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PlacePiece {
    source: Source,
    placement: Coords,
}

impl GameState {
    fn new() -> Self {
        GameState { bank: Bank::Unused, piece: None, tower: Tower::new(), bonus: None }
    }

    fn execute_move(&mut self, mov: &Move) {
        match mov {
            Move::BankPiece => {
                self.bank = Bank::Stored(self.piece.unwrap());
                self.piece = None;
            },
            Move::PlacePiece(place) => {
                match place.source {
                    Source::Bank => self.bank = Bank::Used,
                    Source::Dice => self.piece = None,
                }
                self.bonus = self.tower.set(&place.placement);
            },
        }
    }
    fn score(&self) -> u32 {
        self.tower.count_filled() * self.bonus.unwrap_or(1)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Tower {
    data: Box<[[Tile; 7]; 15]>,
}

impl Tower {
    fn count_filled(&self) -> u32 {
        let mut count = 0;
        for row in (&self.data).iter() {
            for cell in row {
                if *cell == Tile::Filled {
                    count += 1;
                }
            }
        }
        count
    }
    fn get(&self, cell: (usize, usize)) -> Tile {
        self.data[cell.1][cell.0]
    }
    fn try_get(&self, cell: (isize, isize)) -> Tile {
        if (cell.0 > 6) || (cell.0 < 0) | (cell.1 > 14) || (cell.1 < 0) {
            Tile::Null
        } else {
            self.get((cell.0 as usize, cell.1 as usize))
        }
    }
    fn set(&mut self, placement: &Coords) -> Option<u32> {
        let mut bonus = None;
        for value in placement {
            let cell = &mut self.data[value.1 as usize][value.0 as usize];
            if let Tile::Bonus(b) = cell {
                bonus = Some(*b);
            }
            *cell = Tile::Filled;
        }
        bonus
    }
    #[allow(unused)]
    fn print(&self) {
        for row in self.data.iter().rev() {
            for cell in row {
                match cell {
                    Tile::Bonus(b) => print!("{}", b),
                    Tile::Empty => print!("_"),
                    Tile::Filled => print!("X"),
                    Tile::Null => print!(" "),
                }
            }
            println!();
        }
    }
    fn new() -> Self {
        let x = Tile::Null;
        let o = Tile::Empty;
        let a = Tile::Bonus(2);
        let b = Tile::Bonus(3);
        let c = Tile::Bonus(4);
        let d = Tile::Bonus(5);
        Self {
            // Draws the tower "upside down"
            // because I like to think of up as +1 y in the rest of the code.
            data: Box::new([
                [o, o, o, x, o, o, o],
                [o, o, o, o, o, o, o],
                [o, o, o, o, o, o, o],
                [x, o, o, o, o, o, o],
                [x, o, o, o, o, o, o],
                [x, o, o, o, o, o, a],
                [x, o, o, o, o, o, x],
                [x, o, o, o, o, o, x],
                [x, b, o, o, o, o, x],
                [x, x, o, o, o, o, x],
                [x, x, o, o, o, o, x],
                [x, x, o, o, o, c, x],
                [x, x, o, o, o, x, x],
                [x, x, o, o, o, x, x],
                [x, x, o, d, o, x, x],
            ]),
        }
    }
}

fn transform(cell: (isize, isize), amount: &[(isize, isize)]) -> Coords {
    let mut coords = Coords::new_const();
    for a in amount {
        coords.push((cell.0 + a.0, cell.1 + a.1));
    }
    coords
}

fn placement_is_legal(tower: &Tower, placement: &Coords) -> bool {
    for i in 0..placement.len() {
        let cell = placement[i];
        match tower.try_get(cell) {
            Tile::Bonus(_) | Tile::Empty => {
                // Now check the tile below
                let cell_below = (cell.0, cell.1-1);
                // TODO: This is bug if bonus is not at the top.
                if tower.try_get(cell_below) == Tile::Empty {
                    if i == 0 || (placement[i-1]) != cell_below {
                        return false;
                    }
                }
            },
            Tile::Filled | Tile::Null => { return false; }
        }
    }
    true
}


struct Lookup {
    table: Vec<f64>,
    len: usize,
}

impl Lookup {
    fn from_bytes(bytes: &[u8]) -> Self {
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
    fn lookup(&self, index: u32) -> Option<f64> {
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
    fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.table.len());
        for elem in &self.table {
            result.extend(elem.to_le_bytes());
        }
        result
    }
    fn insert(&mut self, index: u32, value: f64) {
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

    fn hash(game: &GameState) -> u32 {
        assert!(game.piece.is_none());
        // TODO: Terrible efficiency. We can do a perfect hash into a u32, affording a
        // Vec instead of giant hashmap
        let mut result = [(0u32, 0u32); 8];
        for row in 0..7 {
            let mut count_filled = 0;
            let mut count_possible = 1;
            for column in 0..15 {
                match game.tower.get((row, column)) {
                    Tile::Filled => {
                        count_filled += 1;
                        count_possible += 1;
                    },
                    Tile::Bonus(_) | Tile::Empty => {
                        count_possible += 1;
                    },
                    Tile::Null => {}
                }
            }
            result[row] = (count_filled, count_possible);
        }
        let bank = match game.bank {
            Bank::Unused => 0,
            Bank::Stored(Piece::FourL) => 1,
            Bank::Stored(Piece::FourRow) => 2,
            Bank::Stored(Piece::FourSquare) => 3,
            Bank::Stored(Piece::FourT) => 4,
            Bank::Stored(Piece::FourZ) => 5,
            Bank::Stored(Piece::One) => 6,
            Bank::Stored(Piece::Two) => 7,
            Bank::Stored(Piece::ThreeL) => 8,
            Bank::Stored(Piece::ThreeRow) => 9,
            Bank::Used => 10,
        };
        result[7] = (bank, 11);
        perfect_hash(&result)
    }
}

fn calculate_ev(game: &GameState, lookup: &mut Lookup) -> f64 {
    if game.piece.is_some() {
        calculate_ev_with_piece(game, lookup)
    } else {
        calculate_ev_no_piece(game, lookup)
    }
}

fn calculate_ev_with_piece(game: &GameState, lookup: &mut Lookup) -> f64 {
    assert!(game.piece.is_some());
    let moves = available_moves(&game);
    if moves.len() == 0 {
        game.score() as f64
    } else {
        let mut max_ev = 0.0f64;
        for mv in moves {
            let mut game = game.clone();
            game.execute_move(&mv);
            let ev = calculate_ev(&game, lookup);
            max_ev = max_ev.max(ev);
        }
        max_ev
    }
}

fn calculate_ev_no_piece(game: &GameState, lookup: &mut Lookup) -> f64 {
    let hash = Lookup::hash(&game);
    if let Some(cached) = lookup.lookup(hash) {
        return cached;
    }
    
    let mut total_score = 0.0;
    let mut total_count = 0.0;
    for (roll, count) in [
        (2, 1.0),
        (3, 2.0),
        (4, 3.0),
        (5, 4.0),
        (6, 4.0),
        (7, 4.0),
        (8, 3.0),
        (9, 2.0),
        (10, 1.0),
    ] {
        let piece = roll_to_piece(roll);
        let mut game = game.clone();
        game.piece = Some(piece);
        let ev = calculate_ev(&game, lookup);
       
        total_count += count;
        total_score += ev * count;
    }
    
    let ev = total_score / total_count;
    lookup.insert(hash, ev);
    ev
}

fn perfect_hash(select_from: &[(u32, u32)]) -> u32 {
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


fn main() {
    // There are less than 147M reachable game states, of ~154M total game states.
    // So, we can calculate perfect play by way of a brute forced a lookup table
    // containing state -> EV and iterate over currently reachable positions to check
    // which is the best.
    /*
    let mut lookup = Lookup {
        len: 0,
        table: Vec::new(),
    };
    
    let game = GameState::new();
    calculate_ev(&game, &mut lookup);
    let data = lookup.to_bytes();
    std::fs::write("./cache.bin", data).unwrap();
    return;
    */
    let mut lookup = Lookup::from_bytes(&std::fs::read("./cache.bin").unwrap());

    let start = Instant::now();
    let mut rng = thread_rng();
    let mut score = 0.0;
    let games = 1;
    let mut high_score = 0;
    for _ in 0..games {
        let mut game = GameState::new();
        loop {
            if game.piece.is_none() {
                game.piece = Some(Piece::random_from_dice(&mut rng));
                println!("{:?}", game.piece.unwrap());
            }
            let moves = available_moves(&game);
            let mut best_move = None;
            let mut best_ev = 0.0;
            for mv in moves {
                let mut game = game.clone();
                game.execute_move(&mv);
                let ev = calculate_ev(&game, &mut lookup);
                print!("{} ", (ev * 100.0).round() / 100.0);
                if ev > best_ev {
                    best_ev = ev;
                    best_move = Some(mv);
                }
            }
            println!();
            if let Some(best_move) = best_move {
                game.execute_move(&best_move);
                
                game.tower.print();
                //println!("{:?}", game.bank);
                //println!("{}", best_ev);
            } else {
                let this_score = game.score();
                if this_score > high_score {
                    high_score = this_score;
                    game.tower.print();
                    println!("{}", game.score());
                }
                score += game.score() as f64;
                //println!("{}", game.score());
                //println!("==========================");
                break;
            }
        }
    }

    dbg!(Instant::now() - start);

    println!("{}", score / games as f64);
    println!("{}", calculate_ev(&GameState::new(), &mut lookup));
    dbg!(lookup.len, lookup.table.len());
}

