


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
    let mut lookup = Lookup::from_bytes(&std::fs::read("./max_cache.bin").unwrap());

    let start = Instant::now();
    let mut rng = thread_rng();
    let mut score = 0.0;
    let games = 1;
    let mut high_score = 0;
    for _ in 0..games {
        let mut game = GameState::new();
        loop {
            // Tested properties:
            // There are meaninful choices, and surprising best moves
            // There is not too much variance for EV on best play from first move
            // There is luck, and scores can swing with perfect play
            
            // Show that there is little variance between rolls on the first move.
            // No game is "bad". The EV are between 268 and 262
            /*
            for roll in 2..11 {
                game.piece = Some(roll_to_piece(roll));
                println!("{:?} {}", game.piece.unwrap(), calculate_ev(&game, &mut lookup));
            }
            return;
            */
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
                if ev > best_ev {
                    best_ev = ev;
                    best_move = Some(mv);
                }
            }
            println!();
            if let Some(best_move) = best_move {
                game.execute_move(&best_move);
                
                game.tower.print();
                println!("{:?}", game.bank);
                //println!("{}", best_ev);
            } else {
                println!("{}", game.score());
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

