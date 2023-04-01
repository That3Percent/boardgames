
use {
    std::{hash::Hash},
    rand::Rng,
    arrayvec::ArrayVec,
};

type Coords = ArrayVec<(isize, isize), 4>;


#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GameState<const W: usize, const H: usize> {
    pub bank: Bank,
    pub piece: Option<Piece>,
    pub tower: Tower<W, H>,
    pub bonus: Option<u32>,
}

impl<const W: usize, const H: usize> GameState<W, H> {
    pub fn available_moves(&self) -> Vec<Move> {
        if self.bonus.is_some() {
            return Vec::new();
        }
        let piece = self.piece.unwrap();

        let mut moves = Vec::new();
        let mut pieces = ArrayVec::<_, 2>::new();

        pieces.push((Source::Dice, piece));

        match self.bank {
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

        for row in 0..W {
            for column in 0..H {
                if self.tower.get((row, column)) == Tile::Empty {
                    let cell = (row as isize, column as isize);
                    for (source, piece) in pieces.iter() {
                        for transformation in piece.transformations() {
                            let placement = transform(cell, &transformation);
                            if self.tower.placement_is_legal(&placement) {
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Bank {
    Unused,
    Stored(Piece),
    Used,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Tile {
    Filled, // Has a piece played
    Empty, // Could have piece played
    Null, // Not a part of the game
    Bonus(u32), // Has a star
}


#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Piece {
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


pub fn roll_to_piece(roll: u32) -> Piece {
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
    pub fn random_from_dice<R: Rng>(r: &mut R) -> Self {
        roll_to_piece(die_roll(r))
    }
    /// Transformations which when evaluated over each placement point yields a unique
    /// and complete set of movement options.
    /// All of them include (0, 0).
    /// Coordinates are (X, Y)
    /// If a tile is directly above another tile, it goes right after the tile it is above.
    pub fn transformations(&self) -> &'static [&'static [(isize, isize)]] {
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


pub fn die_roll<R: Rng>(r: &mut R) -> u32 {
    let d4 = r.gen_range(1..=4);
    let d6 = r.gen_range(1..=6);

    d4 + d6
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Bank,
    Dice,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    BankPiece,
    PlacePiece(PlacePiece),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PlacePiece {
    source: Source,
    placement: Coords,
}

impl<const W: usize, const H: usize> GameState<W, H> {
    pub fn new(tower: Tower<W, H>) -> Self {
        GameState { bank: Bank::Unused, piece: None, tower, bonus: None }
    }

    pub fn execute_move(&mut self, mov: &Move) {
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
    pub fn score(&self) -> u32 {
        self.tower.count_filled() * self.bonus.unwrap_or(1)
    }
}


#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Tower<const W: usize, const H: usize> {
    data: Box<[[Tile; W]; H]>,
}



impl<const W: usize, const H: usize> Tower<W, H> {
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
    fn placement_is_legal(&self, placement: &Coords) -> bool {
        for i in 0..placement.len() {
            let cell = placement[i];
            match self.try_get(cell) {
                Tile::Bonus(_) | Tile::Empty => {
                    // Now check the tile below
                    let cell_below = (cell.0, cell.1-1);
                    // TODO: This is bug if bonus is not at the top.
                    if self.try_get(cell_below) == Tile::Empty {
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
    pub fn get(&self, cell: (usize, usize)) -> Tile {
        self.data[cell.1][cell.0]
    }
    pub fn try_get(&self, cell: (isize, isize)) -> Tile {
        if (cell.0 >= W as isize) || (cell.0 < 0) | (cell.1 >= H as isize) || (cell.1 < 0) {
            Tile::Null
        } else {
            self.get((cell.0 as usize, cell.1 as usize))
        }
    }
    pub fn set(&mut self, placement: &Coords) -> Option<u32> {
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
    pub fn print(&self) {
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
}

pub fn tower() -> Tower<7, 16> {
    let x = Tile::Null;
    let o = Tile::Empty;
    let a = Tile::Bonus(2);
    let b = Tile::Bonus(3);
    let c = Tile::Bonus(4);
    let d = Tile::Bonus(5);
    Tower {
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
            [x, x, o, o, o, x, x],
            [x, x, x, d, x, x, x],
        ]),
    }
}


pub fn wide() -> Tower<11, 7> {
    let x = Tile::Null;
    let o = Tile::Empty;
    Tower {
        // Draws the tower "upside down"
        // because I like to think of up as +1 y in the rest of the code.
        data: Box::new([
            [o, x, x, o, o, x, o, x, x, x, o],
            [o, o, o, o, o, o, o, o, o, o, o],
            [o, o, o, o, o, o, o, o, o, o, o],
            [o, o, o, o, o, o, o, o, o, o, o],
            [o, o, o, o, o, x, o, o, o, o, o],
            [o, o, o, o, o, x, o, x, o, o, o],
            [x, o, o, o, x, x, o, x, x, x, o],
        ]),
    }
}



fn transform(cell: (isize, isize), amount: &[(isize, isize)]) -> Coords {
    let mut coords = Coords::new_const();
    for a in amount {
        coords.push((cell.0 + a.0, cell.1 + a.1));
    }
    coords
}



