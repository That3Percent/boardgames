mod utils;
use utils::*;
mod rules;
use rules::*;

use {
    std::time::Instant,
    rand::thread_rng,
};

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



fn calculate_ev(game: &GameState, lookup: &mut Lookup) -> f64 {
    if game.piece.is_some() {
        calculate_ev_with_piece(game, lookup)
    } else {
        calculate_ev_no_piece(game, lookup)
    }
}

fn calculate_ev_with_piece(game: &GameState, lookup: &mut Lookup) -> f64 {
    assert!(game.piece.is_some());
    let moves = game.available_moves();
    if moves.len() == 0 {
        game.score() as f64
    } else {
        let mut min_ev = f64::MAX;
        for mv in moves {
            let mut game = game.clone();
            game.execute_move(&mv);
            let ev = calculate_ev(&game, lookup);
            min_ev = min_ev.min(ev);
        }
        min_ev
    }
}

fn calculate_ev_no_piece(game: &GameState, lookup: &mut Lookup) -> f64 {
    let hash = hash(&game);
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
            let moves = game.available_moves();
            let mut best_move = None;
            let mut best_ev = f64::MAX;
            for mv in moves {
                let mut game = game.clone();
                game.execute_move(&mv);
                let ev = calculate_ev(&game, &mut lookup);
                if ev < best_ev {
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
}

