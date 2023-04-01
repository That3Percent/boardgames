mod utils;
use utils::*;
mod rules;
use rules::*;

type Num = f32;

use {
    rand::thread_rng,
};

pub trait Strategy: Copy {
    fn worst(&self) -> Num;
    fn prefer(&self, prev: Num, eval: Num) -> bool;
}

#[derive(Copy, Clone)]
pub struct MaxEV;
impl Strategy for MaxEV {
    fn worst(&self) -> Num {
        Num::MIN
    }
    fn prefer(&self, prev: Num, eval: Num) -> bool {
        eval > prev
    }
}

#[derive(Copy, Clone)]
pub struct MinEV;
impl Strategy for MinEV {
    fn worst(&self) -> Num {
        Num::MAX
    }
    fn prefer(&self, prev: Num, eval: Num) -> bool {
        eval < prev
    }
}


fn hash<const W: usize, const H: usize>(game: &GameState<W, H>) -> u32 {
    assert!(game.piece.is_none());
    let mut hasher = PerfectHasher::new();
    for row in 0..W {
        let mut count_filled = 0;
        let mut count_possible = 1;
        for column in 0..H {
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
        hasher.update(count_filled, count_possible);
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
    hasher.update(bank, 11);
    hasher.digest()
}



fn calculate_ev<const W: usize, const H: usize, C: Context>(game: &GameState<W, H>, ctx: &mut C) -> Num {
    if game.piece.is_some() {
        calculate_ev_with_piece(game, ctx)
    } else {
        calculate_ev_no_piece(game, ctx)
    }
}

fn calculate_ev_with_piece<const W: usize, const H: usize, C: Context>(game: &GameState<W, H>, ctx: &mut C) -> Num {
    assert!(game.piece.is_some());
    let moves = game.available_moves();
    if moves.len() == 0 {
        game.score() as Num
    } else {
        let mut best_ev = ctx.strategy().worst();
        for mv in moves {
            let mut game = game.clone();
            game.execute_move(&mv);
            let ev = calculate_ev(&game, ctx);
            if ctx.strategy().prefer(best_ev, ev) {
                best_ev = ev;
            }
        }
        best_ev
    }
}

fn calculate_ev_no_piece<const W: usize, const H: usize, C: Context>(game: &GameState<W, H>, ctx: &mut C) -> Num {
    let hash = hash(&game);
    if let Some(cached) = ctx.lookup().get(&hash) {
        return *cached;
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
        let ev = calculate_ev(&game, ctx);
       
        total_count += count;
        total_score += ev * count;
    }
    
    let ev = total_score / total_count;
    ctx.lookup_mut().insert(hash, ev);
    ev
}

trait Context {
    type Strategy: Strategy;
    type Lookup: Lookup<Key = u32, Value = Num>;
    fn strategy(&self) -> Self::Strategy;
    fn lookup(&self) -> &Self::Lookup;
    fn lookup_mut(&mut self) -> &mut Self::Lookup;
}

struct C<S, L> {
    strategy: S,
    lookup: L,
}

impl<S, L> Context for C<S, L> where S: Strategy, L: Lookup<Key = u32, Value = Num> {
    type Strategy = S;
    type Lookup = L;
    fn strategy(&self) -> Self::Strategy {
        self.strategy
    }
    fn lookup(&self) -> &Self::Lookup {
        &self.lookup
    }
    fn lookup_mut(&mut self) -> &mut Self::Lookup {
        &mut self.lookup
    }
}


fn main() {
    // There are less than 147M reachable game states, of ~154M total game states.
    // So, we can calculate perfect play by way of a brute forced a lookup table
    // containing state -> EV and iterate over currently reachable positions to check
    // which is the best.

    let strategy = MinEV;
    let board = || wide();

    let path = "./min_cache.bin";
    let mut ctx = if let Ok(bytes) = std::fs::read(&path) {
        C {
            lookup: DenseLookup::from_bytes(&bytes),
            strategy,
        }
    } else {
        let mut ctx = C {
            lookup: DenseLookup::new(),
            strategy
        };
        let game = GameState::new(board());
        calculate_ev(&game, &mut ctx);
        let data = ctx.lookup().to_bytes();
        std::fs::write(&path, data).unwrap();
        ctx
    };

    let mut rng = thread_rng();
    let mut score = 0.0;
    let games = 1;
    let mut high_score = 0;
    for _ in 0..games {
        let mut game = GameState::new(board());
        loop {
            if game.piece.is_none() {
                game.piece = Some(Piece::random_from_dice(&mut rng));
                println!("{:?}", game.piece.unwrap());
            }
            let moves = game.available_moves();
            let mut best_move = None;
            let mut best_ev = strategy.worst();
            for mv in moves {
                let mut game = game.clone();
                game.execute_move(&mv);
                let ev = calculate_ev(&game, &mut ctx);
                if strategy.prefer(best_ev, ev) {
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
                score += game.score() as Num;
                //println!("{}", game.score());
                //println!("==========================");
                break;
            }
        }
    }


    println!("{}", score / games as Num);
    println!("{}", calculate_ev(&GameState::new(board()), &mut ctx));
}

