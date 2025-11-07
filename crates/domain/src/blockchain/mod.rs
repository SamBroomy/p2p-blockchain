mod chain;

use blake3::Hash;

use crate::{
    block::Block,
    blockchain::chain::{Chain, ForkChain, RootChain, is_connecting_block_valid},
};

const GENESIS_ROOT_HASH: Hash = Hash::from_bytes([0u8; blake3::OUT_LEN]);

pub struct BlockChain {
    /// The main accepted chain. Should always be the chain with the highest cumulative difficulty
    main_chain: Chain<RootChain>,
    forks: Vec<Chain<ForkChain>>,
    difficulty: usize,
}

impl BlockChain {
    fn create_genesis_block(difficulty: usize) -> Block {
        crate::block::BlockConstructor::new(0, &[], GENESIS_ROOT_HASH).mine(difficulty, None)
    }

    pub fn new(difficulty: usize) -> Self {
        let genesis = Self::create_genesis_block(difficulty);
        Self {
            main_chain: Chain::new_from_genesis(genesis),
            forks: Vec::new(),
            difficulty,
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn add_block(&mut self, block: Block) -> Result<(), Block> {
        // 1. Try to add to the main chain tip
        if let Ok(()) = self.main_chain.add_block(block.clone()) {
            // Block added successfully to main chain
            self.check_for_reorganization();
            self.prune_old_forks();
            return Ok(());
        }
        // 2. Try and add to existing forks
        for fork in &mut self.forks {
            if is_connecting_block_valid(fork.tip_block(), &block)
                && let Ok(()) = fork.add_block(block.clone())
            {
                // Block added to fork successfully
                self.check_for_reorganization();
                self.prune_old_forks();
                return Ok(());
            }
        }
        // 3. Try to create a new fork from main chain
        if self.main_chain.contains_block(block.previous_hash())
            && let Some(mut new_fork) = self.main_chain.truncate_at(block.previous_hash())
        {
            if new_fork.add_block(block.clone()).is_ok() {
                self.forks.push(new_fork);
                self.check_for_reorganization();
                self.prune_old_forks();
                return Ok(());
            }
            // Failed to add to new fork - restore main chain
            self.main_chain.append_fork(new_fork);
        }
        // 4. Try to create a fork from an existing fork
        for i in 0..self.forks.len() {
            if self.forks[i].contains_block(block.previous_hash())
                && let Some(mut new_fork) = self.forks[i].truncate_at(block.previous_hash())
            {
                if new_fork.add_block(block.clone()).is_ok() {
                    self.forks.push(new_fork);
                    self.check_for_reorganization();
                    self.prune_old_forks();
                    return Ok(());
                }
                // Failed - restore original fork
                self.forks[i].append_fork(new_fork);
            }
        }
        // Block cannot be added anywhere - reject
        Err(block)
    }

    fn check_for_reorganization(&mut self) {
        // Find the fork with highest cumulative difficulty
        let main_difficulty = self.main_chain.cumulative_difficulty();

        let mut best_fork_idx = None;
        let mut best_difficulty = main_difficulty;

        for (idx, fork) in self.forks.iter().enumerate() {
            if !fork.valid_fork(&self.main_chain) {
                continue;
            }

            // Calculate total difficulty if this fork were to become main
            let fork_point_idx = self
                .main_chain
                .find_block_index(fork.root_hash())
                .expect("valid fork must connect to main chain");

            let root_difficulty = self.main_chain.cumulative_difficulty_till(fork_point_idx);
            let fork_difficulty = fork.cumulative_difficulty();
            let total_difficulty = root_difficulty + fork_difficulty;

            if total_difficulty > best_difficulty {
                best_difficulty = total_difficulty;
                best_fork_idx = Some(idx);
            }
        }

        // Perform reorganization if needed
        if let Some(fork_idx) = best_fork_idx {
            let winning_fork = self.forks.remove(fork_idx);

            match self.main_chain.merge_fork(winning_fork) {
                Ok(old_main_chain_fork) => {
                    // The old main chain tail is now a fork
                    self.forks.push(old_main_chain_fork);
                    println!("Chain reorganization occurred!");
                }
                Err(fork) => {
                    // Merge failed (shouldn't happen), restore fork
                    self.forks.insert(fork_idx, fork);
                }
            }
        }
    }

    fn prune_old_forks(&mut self) {
        const MAX_FORK_DEPTH: usize = 10;
        let main_len = self.main_chain.len();

        self.forks.retain(|fork| {
            self.main_chain
                .find_block_index(fork.root_hash())
                .is_some_and(|idx| main_len - idx <= MAX_FORK_DEPTH)
        });
    }
}
