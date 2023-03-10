use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

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



fn eval_position(game_state: &mut GameState) -> f64 {
    const COUNT: usize = 32;
    let tests = vec![0; COUNT];
    return tests.par_iter().map(|_| {
        let mut game_state = game_state.clone();
        let mut rng = thread_rng();
        loop {
            if game_state.piece.is_none() {
                game_state.piece = Some(Piece::random_from_dice(&mut rng));
            }

            let available_moves = available_moves(&game_state);
            if available_moves.len() == 0 {
                return game_state.score() as f64;
            }
            let choice = rng.gen_range(0..available_moves.len());
            for (i, mv) in available_moves.into_iter().enumerate() {
                if i == choice {
                    game_state.execute_move(&mv);
                    break;
                }
            }
        }
    }).sum::<f64>() / COUNT as f64;
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
            Move::BankPiece => {self.bank = Bank::Stored(self.piece.unwrap()); },
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
                    Tile::Empty => print!("□"),
                    Tile::Filled => print!("▪️"),
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



#[derive(Debug)]
struct MCTSScore {
    visits: f64,
    score: f64,
}


#[derive(Debug)]
enum MCTSData {
    Null,
    Dice(Box<[u32; 9]>),
    Move(Vec<(Move, u32)>),
}

impl MCTSData {
    fn len(&self) -> usize {
        match self {
            MCTSData::Dice(d) => d.len(),
            MCTSData::Move(m) => m.len(),
            MCTSData::Null => 0,
        }
    }
}

struct MCTSContext {
    visit: Vec<u32>,
    game: GameState,
    nodes: Vec<MCTSData>,
    scores: Vec<MCTSScore>
}

fn mcts_expand2(context: &mut MCTSContext, id: u32) {
    let data = if context.game.piece.is_none() {
        MCTSData::Dice(Box::new([0; 9]))
    } else {
        let mut moves = Vec::new();
        for mv in available_moves(&mut context.game) {
            moves.push((mv, 0));
        }
        MCTSData::Move(moves)
    };
    context.nodes[id as usize] = data;
}

fn mcts_expand(context: &mut MCTSContext) {
    context.nodes.push(MCTSData::Null);
    context.scores.push(MCTSScore { visits: 0.0, score: 0.0 })
}

fn visit_score(node: u32, nodes: &[MCTSScore]) -> f64 {
    if node != 0 {
        let node = &nodes[node as usize];
        node.score / node.visits
    } else { 500.0 }
}

fn mcts_select<R: Rng>(context: &mut MCTSContext, r: &mut R) {
    let mut node = 0;

    loop {
        context.visit.push(node);
        let next = context.nodes.len();
        match &mut context.nodes[node as usize] {
            MCTSData::Null => {
                mcts_expand2(context, node);
                context.visit.pop();
                continue;
            }
            MCTSData::Dice(data) => {
                let roll = die_roll(r);
                let piece = roll_to_piece(roll);
                context.game.piece = Some(piece);
                let entry = &mut data[roll as usize - 2];
                if *entry == 0 {
                    *entry = next as u32;
                    context.visit.push(*entry);
                    mcts_expand(context);
                    break;
                }
                node = *entry;
            },
            MCTSData::Move(moves) => {
                if moves.len() == 0 {
                    break;
                }
                let mut total = moves.len() as f64;
                let mut min = f64::MAX;
                for mv in moves.iter() {
                    let score = visit_score(mv.1, &context.scores);
                    total += score;
                    if score < min {
                        min = score;
                    }
                }
                total -= min * moves.len() as f64;
                let r = r.gen_range(0.0..total);
                let mut n = 0;
                let mut total = 0.0;
                while total < r {
                    total += 1.0 + visit_score(moves[n].1, &context.scores)- min;
                    n += 1;
                }
                let entry = &mut moves[n-1];
                context.game.execute_move(&entry.0);
                if entry.1 == 0 {
                    entry.1 = next as u32;
                    context.visit.push(entry.1);
                    mcts_expand(context);
                    break;
                }
                node = entry.1;
            }
        }
    }
}

fn mcts_run<R: Rng>(context: &mut MCTSContext, r: &mut R) {
    context.visit.clear();
    mcts_select(context, r);
    let score = eval_position(&mut context.game);
    for n in context.visit.iter() {
        let node = &mut context.scores[*n as usize];
        node.visits += 1.0;
        node.score += score as f64;
    }
}


fn main() {
    let mut r = rand::thread_rng();
    let mut context = MCTSContext {
        visit: Vec::new(),
        game: GameState::new(),
        nodes: Vec::new(),
        scores: Vec::new(),
    };
    loop {
        let mut game = GameState::new();
        
        loop {
            if game.piece.is_none() {
                game.piece = Some(Piece::random_from_dice(&mut r));
            }
            println!("{:?}", game.piece.unwrap());
            context.game = game.clone();
            context.nodes.clear();
            context.scores.clear();
            mcts_expand(&mut context);
            mcts_expand2(&mut context, 0);
            if context.nodes[0].len() == 0 {
                break;
            }
            match context.nodes[0].len() {
                0 => break,
                1 => {},
                _ => {
                    for _ in 0..20_000_000 {
                        context.game = game.clone();
                        mcts_run(&mut context, &mut r);
                    }
                }
            }
            // Debug visit counts
            /*
            if let MCTSData::Move(moves) = &context.nodes[0] {
                for mv in moves {
                    let score = &context.scores[mv.1 as usize];
                    println!("{}:{}", score.visits, score.score / score.visits);
                }
            }
            return;
            */

            let mut best_score = 0.0;
            let mut best_move = None;
            if let MCTSData::Move(moves) = &context.nodes[0] {
                for i in 0..moves.len() {
                    let score = visit_score(moves[i].1, &context.scores);

                    if score > best_score {
                        best_score = score;
                        best_move = Some(moves[i].0.clone());
                    }
                }
            }

            game.execute_move(&best_move.unwrap());
            game.tower.print();
            println!("{:?}", game.bank);

        }
        println!("{}", game.score());
        println!("=============================");
        //return;
    }

    
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tower() {
        let tower = Tower::new();

        assert!(tower.try_get((3, 0)) == Tile::Null);
        assert!(tower.try_get((-1, -1)) == Tile::Null);
        assert!(tower.try_get((6, 5)) == Tile::Bonus(2));
    }
}