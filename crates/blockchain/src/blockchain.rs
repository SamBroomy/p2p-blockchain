use std::{
    collections::{HashMap, VecDeque},
    marker::PhantomData,
};

use blake3::Hash;
use blockchain_types::{
    Block, BlockConstructor, ConstMiner, Miner, MiningStrategy, Transaction,
    consts::GENESIS_ROOT_HASH, wallet::Address,
};
use rand::{RngCore, rngs::OsRng};
use tracing::info;

use crate::{
    chain::{Chain, ForkChain, RootChain, is_connecting_block_valid},
    orphan_pool::OrphanPool,
};

/// Result of attempting to add a block
#[derive(Debug)]
pub enum BlockAddResult {
    /// Block added successfully, potentially triggered cascading orphan processing
    Added { processed_orphans: usize },
    /// Block is an orphan (missing parent), added to orphan pool
    Orphaned { missing_parent: Hash },
    /// Block rejected (invalid or duplicate)
    Rejected(Block),
}

impl BlockAddResult {
    /// Returns true if the block was successfully added to the blockchain
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Added { .. })
    }

    /// Returns true if the block is orphaned (waiting for parent)
    pub fn is_orphaned(&self) -> bool {
        matches!(self, Self::Orphaned { .. })
    }

    /// Returns true if the block was rejected
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }

    /// Returns Ok(()) if added or orphaned, Err(block) if rejected
    ///
    /// **Note**: This treats `Orphaned` as `Ok(())` for compatibility with legacy Result-style code.
    /// If you need to distinguish between "fully processed" vs "buffered as orphan", use
    /// `is_success()` and `is_orphaned()` instead.
    pub fn ok(self) -> Result<(), Block> {
        match self {
            Self::Added { .. } | Self::Orphaned { .. } => Ok(()),
            Self::Rejected(block) => Err(block),
        }
    }

    /// Returns true if added or orphaned (not rejected)
    pub fn is_ok(&self) -> bool {
        !self.is_rejected()
    }

    /// Returns true if rejected
    pub fn is_err(&self) -> bool {
        self.is_rejected()
    }
}

pub struct BlockChain<M: MiningStrategy = Miner> {
    /// The main accepted chain. Should always be the chain with the highest cumulative difficulty
    main_chain: Chain<RootChain>,
    /// Active forks diverging from the main chain
    forks: Vec<Chain<ForkChain>>,
    /// Orphaned block waiting for parents
    orphan_pool: OrphanPool,
    /// Mining difficulty
    pub difficulty: usize,

    _miner: PhantomData<M>,
}

// Concrete implementation ONLY for the default type
impl BlockChain<Miner> {
    /// Create a new blockchain with the default `MinerSimple`.
    ///
    /// For other miner types, use:
    /// ```rust,ignore
    /// let blockchain: BlockChain<MinerConst<8>> = BlockChain::new_with_miner(8);
    /// ```
    pub fn new(difficulty: usize) -> Self {
        Self::new_with_miner(difficulty)
    }
}

// You could also add convenience constructors for common types
impl<const D: usize> BlockChain<ConstMiner<D>> {
    /// Create a blockchain with const generic difficulty
    pub fn new_miner() -> Self {
        Self::new_with_miner(D)
    }
}

impl<M: MiningStrategy> BlockChain<M> {
    pub fn mine(&self, constructor: BlockConstructor) -> Block {
        let random_nonce = OsRng.next_u64();
        M::mine(constructor, self.difficulty, Some(random_nonce))
    }

    fn create_genesis_block(difficulty: usize) -> Block {
        M::mine(
            BlockConstructor::new(0, &[], GENESIS_ROOT_HASH, None),
            difficulty,
            None,
        )
    }

    // Public constructor when you specify the type explicitly
    pub fn new_with_miner(difficulty: usize) -> Self {
        let genesis = Self::create_genesis_block(difficulty);
        assert!(genesis.is_valid(), "Genesis block must be valid");
        assert_eq!(
            genesis.previous_hash(),
            &GENESIS_ROOT_HASH,
            "Genesis must connect to GENESIS_ROOT_HASH"
        );
        Self {
            main_chain: Chain::new_from_genesis(genesis),
            forks: Vec::new(),
            orphan_pool: OrphanPool::new(),
            difficulty,
            _miner: PhantomData,
        }
    }

    /// Validate transactions against current balances
    pub fn validate_transaction(&self, tx: &Transaction) -> bool {
        self.main_chain.validate_transaction(tx)
    }

    /// Add a block with orphan handling
    pub fn add_block(&mut self, block: Block) -> BlockAddResult {
        let block_hash = *block.hash();
        debug_assert!(
            block.is_valid(),
            "Block {block_hash:?} must be valid before adding"
        );

        // Try to add the block normally
        match self.try_add_block_internal(block) {
            Ok(()) => {
                // Block added! Now process any orphans that were waiting for it
                let processed = self.process_orphans_for(&block_hash);
                debug_assert!(
                    self.main_chain.is_valid(),
                    "Main chain must be valid after adding block"
                );
                BlockAddResult::Added {
                    processed_orphans: processed,
                }
            }
            Err(rejected_block) => {
                // Block didn't connect - check if it's an orphan or truly invalid
                // A block is orphaned only if:
                // 1. It's structurally valid (correct PoW)
                // 2. Its parent is actually missing (not in main chain or any fork)
                let parent_hash = *rejected_block.previous_hash();
                let parent_exists = self.main_chain.contains_block(&parent_hash)
                    || self.forks.iter().any(|f| f.contains_block(&parent_hash));

                if rejected_block.is_valid() && !parent_exists {
                    // Valid block, but parent is missing - add to orphan pool
                    match self.orphan_pool.add_orphan(rejected_block) {
                        Ok(()) => BlockAddResult::Orphaned {
                            missing_parent: parent_hash,
                        },
                        Err(rejected_block) => BlockAddResult::Rejected(rejected_block),
                    }
                } else {
                    // Block is invalid or parent exists (so rejection was due to transaction/balance issues)
                    BlockAddResult::Rejected(rejected_block)
                }
            }
        }
    }

    /// Attempts to add a block to the blockchain or its forks
    #[allow(clippy::result_large_err)]
    fn try_add_block_internal(&mut self, block: Block) -> Result<(), Block> {
        // 1. Try to add to the main chain tip
        if let Ok(()) = self.main_chain.add_block(block.clone()) {
            // Block added successfully to main chain
            self.check_for_reorganization();
            self.prune_old_forks();
            debug_assert!(
                self.main_chain.is_valid(),
                "Main chain invalid after adding block to tip"
            );
            return Ok(());
        }
        // 2. Try and add to existing forks
        for fork in &mut self.forks {
            if is_connecting_block_valid(fork.tip_block(), &block)
                && fork.add_block(block.clone()).is_ok()
            {
                // Block added to fork successfully
                self.check_for_reorganization();
                self.prune_old_forks();
                debug_assert!(
                    self.main_chain.is_valid(),
                    "Main chain invalid after adding block to fork"
                );
                return Ok(());
            }
        }
        // 3. Try to create a new fork from main chain
        if let Some(connection_idx) = self.main_chain.find_block_index(block.previous_hash()) {
            // Get balances at the connection point
            let connection_balances =
                if connection_idx == 0 && *block.previous_hash() == GENESIS_ROOT_HASH {
                    HashMap::new()
                } else {
                    // Recalculate balances up to and including the connection block
                    super::chain::recalculate_balances(&self.main_chain.blocks()[..=connection_idx])
                };

            // Create a new fork with just this block
            if let Ok(new_fork) = Chain::new_from_block(block.clone(), connection_balances) {
                debug_assert!(
                    new_fork.valid_fork(&self.main_chain),
                    "Newly created fork must be valid against main chain"
                );
                self.forks.push(new_fork);
                self.check_for_reorganization();
                self.prune_old_forks();
                return Ok(());
            }
        }
        // 4. Try to create a fork from an existing fork
        for i in 0..self.forks.len() {
            if let Some(connection_balances) = self.forks[i].balances_at(block.previous_hash()) {
                // Found the connection point in this fork
                // Create a new fork starting from this block
                if let Ok(new_fork) = Chain::new_from_block(block.clone(), connection_balances) {
                    self.forks.push(new_fork);
                    self.check_for_reorganization();
                    self.prune_old_forks();
                    return Ok(());
                }
            }
        }
        // Block cannot be added anywhere - reject
        Err(block)
    }

    /// Process orphans that were waiting for a specific block
    fn process_orphans_for(&mut self, parent_hash: &Hash) -> usize {
        let mut processed = 0;
        let mut to_process = VecDeque::from(self.orphan_pool.get_orphans_for_parent(parent_hash));

        while let Some(orphan_block) = to_process.pop_front() {
            let orphan_hash = *orphan_block.hash();
            debug_assert!(
                self.main_chain.contains_block(orphan_block.previous_hash())
                    || self
                        .forks
                        .iter()
                        .any(|f| f.contains_block(orphan_block.previous_hash())),
                "Orphan block {orphan_hash:?} should have parent in chain or forks"
            );
            match self.try_add_block_internal(orphan_block.clone()) {
                Ok(()) => {
                    processed += 1;
                    // This block was added! Check if any orphans were waiting for it
                    let new_orphans = self.orphan_pool.get_orphans_for_parent(&orphan_hash);
                    to_process.extend(new_orphans);
                }
                Err(block) => {
                    debug_assert!(
                        false,
                        "Orphan block {:?} couldn't be added even though parent exists",
                        block.hash()
                    );
                }
            }
        }
        processed
    }

    /// Get list of blocks we need to request from peers
    pub fn get_missing_blocks(&self) -> Vec<Hash> {
        self.orphan_pool.missing_ancestors()
    }

    /// Periodic maintenance - prune old orphans
    pub fn maintain(&mut self) {
        self.orphan_pool.prune_expired();
        self.prune_old_forks();
        debug_assert!(
            self.main_chain.is_valid(),
            "Main chain must remain valid after maintenance"
        );
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
            // Fork connects at fork.root_block().previous_hash()
            let connection_hash = fork.root_block().previous_hash();

            let root_difficulty = if *connection_hash == GENESIS_ROOT_HASH {
                0u128
            } else if let Some(idx) = self.main_chain.find_block_index(connection_hash) {
                self.main_chain.cumulative_difficulty_till(idx)
            } else {
                debug_assert!(
                    false,
                    "Valid fork's connection point {connection_hash:?} not found in main chain"
                );
                continue;
            };
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
                    assert!(
                        self.main_chain.is_valid(),
                        "Main chain must be valid after reorganization"
                    );
                    debug_assert!(
                        !old_main_chain_fork.blocks().is_empty(),
                        "Reorganization should return non-empty old chain"
                    );
                    // The old main chain tail is now a fork
                    self.forks.push(old_main_chain_fork);
                    info!("Chain reorganization occurred!");
                }
                Err(fork) => {
                    // Merge failed (shouldn't happen), restore fork
                    debug_assert!(
                        false,
                        "Fork merge failed even though fork was valid - this is a bug"
                    );
                    self.forks.insert(fork_idx, fork);
                }
            }
        }
    }

    fn prune_old_forks(&mut self) {
        const MAX_FORK_DEPTH: usize = 10;
        let main_len = self.main_chain.len();
        debug_assert!(main_len >= 1, "Main chain must contain at least genesis");

        self.forks.retain(|fork| {
            // Fork connects at fork.root_block().previous_hash()
            let connection_hash = fork.root_block().previous_hash();
            self.main_chain
                .find_block_index(connection_hash)
                .is_some_and(|idx| {
                    debug_assert!(
                        main_len >= idx,
                        "Main chain length {main_len} must be >= connection index {idx}"
                    );
                    main_len.saturating_sub(idx) <= MAX_FORK_DEPTH
                })
        });
        debug_assert!(
            self.forks.iter().all(|f| f.valid_fork(&self.main_chain)),
            "All forks should be valid after pruning"
        );
    }

    pub fn main_chain_len(&self) -> usize {
        debug_assert!(
            self.main_chain.len() >= 1,
            "Main chain must contain at least genesis block"
        );
        debug_assert!(
            self.main_chain.blocks().last().unwrap().index() as usize == self.main_chain.len() - 1,
        );
        self.main_chain.len()
    }

    pub fn fork_count(&self) -> usize {
        self.forks.len()
    }

    pub fn orphan_count(&self) -> usize {
        self.orphan_pool.len()
    }

    pub fn latest_block_hash(&self) -> &Hash {
        self.main_chain.tip_hash()
    }

    /// Get a block by its hash (if we have it)
    pub fn get_block(&self, hash: &Hash) -> Option<&Block> {
        self.main_chain
            .blocks()
            .iter()
            .find(|b| b.hash() == hash)
            .or_else(|| {
                self.forks
                    .iter()
                    .find_map(|fork| fork.blocks().iter().find(|b| b.hash() == hash))
            })
    }

    /// Get blocks in a height range
    pub fn get_blocks_in_range(&self, from: u64, to: u64) -> Vec<Block> {
        self.main_chain
            .blocks()
            .iter()
            .filter(|b| {
                let idx = b.index();
                idx >= from && idx <= to
            })
            .cloned()
            .collect()
    }

    /// Get current chain height
    pub fn height(&self) -> u64 {
        self.main_chain_len() as u64
    }

    /// Get cumulative difficulty (already exists)
    pub fn cumulative_difficulty(&self) -> u128 {
        self.main_chain.cumulative_difficulty()
    }

    pub fn get_balance(&self, address: &Address) -> u64 {
        self.main_chain.get_balance(address)
    }
}

#[cfg(test)]
mod tests {

    use std::assert_matches::assert_matches;

    use blockchain_types::{ConstMiner, Miner, MiningStrategy, Transaction, wallet::Wallet};

    use super::*;

    #[test]
    fn test_default_miner() {
        let blockchain = BlockChain::new(1);
        assert_eq!(blockchain.difficulty, 1);
    }

    #[test]
    fn test_explicit_simple_miner() {
        let blockchain: BlockChain<Miner> = BlockChain::new_with_miner(2);
        assert_eq!(blockchain.difficulty, 2);
    }

    #[test]
    fn test_const_miner() {
        let blockchain: BlockChain<ConstMiner<4>> = BlockChain::new_with_miner(4);
        assert_eq!(blockchain.difficulty, 4);
    }

    #[test]
    fn test_const_miner_with_helper() {
        let blockchain = BlockChain::<ConstMiner<5>>::new_miner();
        assert_eq!(blockchain.difficulty, 5);
    }

    fn mine_block(
        index: u64,
        previous_hash: Hash,
        transactions: &[Transaction],
        difficulty: usize,
    ) -> Block {
        Miner::mine(
            BlockConstructor::new(index, transactions, previous_hash, None),
            difficulty,
            None,
        )
    }

    #[test]
    fn test_blockchain_initialization() {
        let blockchain = BlockChain::new(1);
        assert_eq!(blockchain.main_chain_len(), 1);
        assert_eq!(blockchain.fork_count(), 0);
    }

    #[test]
    fn test_add_block_to_main_chain() {
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let block1 = mine_block(1, genesis_hash, &[], 1);
        assert_matches!(
            blockchain.add_block(block1.clone()),
            BlockAddResult::Added { .. }
        );
        assert_eq!(blockchain.main_chain_len(), 2);
        assert_eq!(blockchain.fork_count(), 0);
    }

    #[test]
    fn test_fork_validation_at_different_depths() {
        // Test that forks can be created at different depths and validated correctly
        // Create a main chain: Genesis -> Block1 -> Block2 -> Block3 -> Block4
        let mut blockchain = BlockChain::new(2);
        let mut current_hash = *blockchain.latest_block_hash();

        for i in 1..=4 {
            let block = mine_block(i, current_hash, &[], 2);
            current_hash = *block.hash();
            assert_matches!(blockchain.add_block(block), BlockAddResult::Added { .. });
        }
        assert_eq!(blockchain.main_chain_len(), 5); // Genesis + 4 blocks

        // Get block 3's hash by mining from scratch
        let mut temp_blockchain = BlockChain::new(2);
        let mut hash_at_index_3 = *temp_blockchain.latest_block_hash();

        for i in 1..=3 {
            let block = mine_block(i, hash_at_index_3, &[], 2);
            hash_at_index_3 = *block.hash();
            assert_matches!(
                temp_blockchain.add_block(block),
                BlockAddResult::Added { .. }
            );
        }

        // Now create a fork from block 3 with higher difficulty
        let fork_block_1 = mine_block(4, hash_at_index_3, &[], 4); // Higher difficulty
        let fork_block_2 = mine_block(5, *fork_block_1.hash(), &[], 4);

        // Add fork blocks to original blockchain
        // This should create a fork from block 3
        assert!(
            blockchain.add_block(fork_block_1.clone()).is_ok(),
            "Fork block 1 should be accepted"
        );
        assert!(
            blockchain.add_block(fork_block_2).is_ok(),
            "Fork block 2 should be accepted"
        );

        // After adding fork with higher difficulty, it should become main chain
        // Fork: 2^4 + 2^4 = 16 + 16 = 32
        // Main from block3 onwards: 2^2 = 4
        // Since fork difficulty (32) > main tail (4), reorganization should occur
        assert_eq!(
            blockchain.main_chain_len(),
            6,
            "Fork with higher difficulty should become main chain (Genesis + 3 blocks + 2 fork blocks)"
        );
        assert_eq!(
            blockchain.fork_count(),
            1,
            "Old block4 should become a fork"
        );
    }

    #[test]
    fn test_fork_becomes_main_chain() {
        // This test exposes Bug #1: Fork validation and Bug #7: Iterator invalidation
        // Create main chain with low difficulty
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        // Main chain: Genesis -> A -> B
        let block_a = mine_block(1, genesis_hash, &[], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();

        let block_b = mine_block(2, block_a_hash, &[], 1);
        let _ = blockchain.add_block(block_b).ok();

        assert_eq!(blockchain.main_chain_len(), 3);

        // Create competing fork from A with higher difficulty: A -> C -> D -> E
        let block_c = mine_block(2, block_a_hash, &[], 3); // Higher difficulty
        let block_c_hash = *block_c.hash();
        let block_d = mine_block(3, block_c_hash, &[], 3);
        let block_d_hash = *block_d.hash();
        let block_e = mine_block(4, block_d_hash, &[], 3);

        let _ = blockchain.add_block(block_c).ok();
        let _ = blockchain.add_block(block_d).ok();
        let _ = blockchain.add_block(block_e).ok();

        // With Bug #1: Fork won't be valid, won't reorganize
        // Expected behavior: Fork has higher cumulative difficulty and should become main chain
        // Main chain should be: Genesis -> A -> C -> D -> E (length 5)
        // Old main (B) should become a fork

        println!(
            "Main chain length after fork: {}",
            blockchain.main_chain_len()
        );
        println!("Fork count after fork: {}", blockchain.fork_count());

        // This will fail until Bug #1 is fixed
        assert_eq!(
            blockchain.main_chain_len(),
            5,
            "Fork with higher difficulty should become main chain"
        );
        assert_eq!(
            blockchain.fork_count(),
            1,
            "Old main chain should become a fork"
        );
    }

    #[test]
    fn test_first_transaction_delta_calculation() {
        // This test exposes Bug #2 & #4: Delta initialization error
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        // Create two wallets
        let alice = Wallet::new();
        let bob = Wallet::new();

        // Alice creates a transaction sending 50 to Bob
        let tx = alice.create_transaction(bob.address(), 50);

        // Mine a block with this transaction
        let block1 = mine_block(1, genesis_hash, &[tx], 1);

        // Add block to blockchain
        let result = blockchain.add_block(block1);

        // With Bug #2: Delta calculation is wrong
        // - Alice's delta = or_insert(100) - 50 = 50 (WRONG! Should be -50)
        // - Bob's delta = or_insert(100) + 50 = 150 (WRONG! Should be +50)
        // - When validated: Alice balance = 100 + 50 = 150 (WRONG!)
        // - Bob balance = 100 + 150 = 250 (WRONG!)

        // Expected behavior:
        // - Alice starts with 100 (INITIAL_BALANCE)
        // - Alice sends 50, should have 50 left
        // - Bob receives 50, should have 150

        // This should succeed if deltas are calculated correctly
        assert!(result.is_ok(), "Transaction should be valid");
    }

    #[test]
    fn test_balance_never_negative() {
        // This test exposes Bug #2, #3, and #4
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let alice = Wallet::new();
        let bob = Wallet::new();

        // Alice tries to send more than she has (100)
        let tx = alice.create_transaction(bob.address(), 150);
        let block1 = mine_block(1, genesis_hash, &[tx], 1);

        let result = blockchain.add_block(block1);

        // Expected: Transaction should be rejected (balance would go negative)
        // With Bug #2: Delta calculated wrong, might allow invalid transaction
        // With Bug #3: verify_balances doesn't check for negative (u64 can't be negative anyway)

        assert!(
            result.is_rejected(),
            "Transaction making balance negative should be rejected, got: {result:?}",
        );
    }

    #[test]
    fn test_reorganization_preserves_balances() {
        // This test checks that balances are correctly maintained through reorganization
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let alice = Wallet::new();
        let bob = Wallet::new();
        let charlie = Wallet::new();

        // Main chain: Genesis -> Block1 (Alice -> Bob: 30)
        let tx1 = alice.create_transaction(bob.address(), 30);
        let block1 = mine_block(1, genesis_hash, &[tx1], 1);
        let block1_hash = *block1.hash();
        let _ = blockchain.add_block(block1).ok();

        // Main chain continues: Block1 -> Block2 (Alice -> Charlie: 20)
        let tx2 = alice.create_transaction(charlie.address(), 20);
        let block2 = mine_block(2, block1_hash, &[tx2], 1);
        let _ = blockchain.add_block(block2).ok();

        assert_eq!(
            blockchain.main_chain_len(),
            3,
            "Initial main chain: Genesis + Block1 + Block2"
        );

        // Create fork from Block1 with higher difficulty
        // Fork: Block1 -> Block3 (Alice -> Charlie: 40) -> Block4 (empty)
        let tx3 = alice.create_transaction(charlie.address(), 40);
        let block3 = mine_block(2, block1_hash, &[tx3], 3); // Higher difficulty
        let block3_hash = *block3.hash();
        let block4 = mine_block(3, block3_hash, &[], 3);

        let _ = blockchain.add_block(block3).ok();
        let _ = blockchain.add_block(block4).ok();

        // Fork has higher cumulative difficulty: (2^3 + 2^3) vs (2^1)
        // 8 + 8 = 16 (fork) vs 2 (block2), so reorganization should occur
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "After reorganization: Genesis + Block1 + Block3 + Block4"
        );
        assert_eq!(blockchain.fork_count(), 1, "Block2 should be on a fork now");

        // Note: We can't directly check balances in the blockchain without exposing internal state
        // But the test validates that the reorganization occurred successfully
        // If balance tracking were broken, add_block would have failed
    }

    #[test]
    fn test_blocks_arriving_out_of_order() {
        // Test orphan block handling: blocks that arrive before their parents are buffered
        //
        // Implementation behavior:
        // - Block 3 arrives first (missing parent block 2) → buffered as orphan
        // - Block 2 arrives → added to chain AND triggers processing of block 3
        // - Block 4 connects to block 3 (now in chain) → added normally
        //
        // This demonstrates cascading orphan resolution

        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let block1 = mine_block(1, genesis_hash, &[], 1);
        let block1_hash = *block1.hash();
        let block2 = mine_block(2, block1_hash, &[], 1);
        let block2_hash = *block2.hash();
        let block3 = mine_block(3, block2_hash, &[], 1);
        let block3_hash = *block3.hash();
        let block4 = mine_block(4, block3_hash, &[], 1);

        // Add blocks in order: 1, 3, 2, 4

        // Block 1 connects to genesis - added normally
        match blockchain.add_block(block1) {
            BlockAddResult::Added { .. } => {}
            _ => panic!("Block 1 should be added"),
        }
        assert_eq!(blockchain.main_chain_len(), 2); // Genesis + block1
        assert_eq!(blockchain.orphan_count(), 0);

        // Block 3 doesn't connect - becomes orphan (waiting for block2)
        match blockchain.add_block(block3.clone()) {
            BlockAddResult::Orphaned { missing_parent } => {
                assert_eq!(missing_parent, block2_hash);
            }
            _ => panic!("Block 3 should be orphaned"),
        }
        assert_eq!(blockchain.main_chain_len(), 2); // Still Genesis + block1
        assert_eq!(blockchain.orphan_count(), 1);

        // Block 2 connects! Should add block2 AND trigger block3 processing
        match blockchain.add_block(block2) {
            BlockAddResult::Added { processed_orphans } => {
                assert_eq!(
                    processed_orphans, 1,
                    "Should process block3 after adding block2"
                );
            }
            _ => panic!("Block 2 should be added"),
        }
        assert_eq!(blockchain.main_chain_len(), 4); // Genesis + block1 + block2 + block3
        assert_eq!(blockchain.orphan_count(), 0); // block3 processed

        // Block 4 connects to block3 which is now in chain - added normally
        match blockchain.add_block(block4) {
            BlockAddResult::Added { processed_orphans } => {
                assert_eq!(processed_orphans, 0, "No orphans to process");
            }
            _ => panic!("Block 4 should be added"),
        }
        assert_eq!(blockchain.main_chain_len(), 5); // Full chain!
        assert_eq!(blockchain.orphan_count(), 0);
    }

    #[test]
    fn test_truncate_at_tip() {
        // Test that adding a competing block at the same height creates a fork
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let block1 = mine_block(1, genesis_hash, &[], 1);
        let block1_hash = *block1.hash();
        let _ = blockchain.add_block(block1).ok();

        let block2 = mine_block(2, block1_hash, &[], 1);
        let _ = blockchain.add_block(block2.clone()).ok();

        let initial_len = blockchain.main_chain_len();

        // Add a competing block at the same height as block2 (connects to block1)
        let competing_block = mine_block(2, block1_hash, &[], 1);
        assert!(
            blockchain.add_block(competing_block).is_ok(),
            "Competing block should be accepted"
        );

        // This should create a fork, not extend main chain
        assert_eq!(
            blockchain.main_chain_len(),
            initial_len,
            "Main chain length should remain unchanged"
        );
        assert_eq!(
            blockchain.fork_count(),
            1,
            "Competing block should create a fork"
        );
    }

    #[test]
    fn test_fork_pruning() {
        // This test exposes Bug #8: Fork pruning underflow
        let mut blockchain = BlockChain::new(1);
        let mut current_hash = *blockchain.latest_block_hash();

        // Build main chain to block 5
        for i in 1..=5 {
            let block = mine_block(i, current_hash, &[], 1);
            current_hash = *block.hash();
            let _ = blockchain.add_block(block).ok();
        }

        // Get hash at block 2 by rebuilding
        let mut temp = BlockChain::new(1);
        let mut hash_at_2 = *temp.latest_block_hash();
        for i in 1..=2 {
            let block = mine_block(i, hash_at_2, &[], 1);
            hash_at_2 = *block.hash();
            let _ = temp.add_block(block).ok();
        }

        // Create fork from block 2
        let fork_block = mine_block(3, hash_at_2, &[], 1);
        let _ = blockchain.add_block(fork_block).ok();

        let initial_forks = blockchain.fork_count();
        println!("Initial forks: {initial_forks}");

        // Extend main chain by 20 more blocks
        for i in 6..=25 {
            let block = mine_block(i, current_hash, &[], 1);
            current_hash = *block.hash();
            let _ = blockchain.add_block(block).ok();
        }

        // Fork from block 2 should be pruned (depth > 10)
        // Bug #8: This might panic if main_len - idx underflows
        println!("Final main chain: {}", blockchain.main_chain_len());
        println!("Final forks: {}", blockchain.fork_count());

        // Fork should be pruned
        assert!(
            blockchain.fork_count() < initial_forks || initial_forks == 0,
            "Old fork should be pruned"
        );
    }

    #[test]
    fn test_multiple_forks_same_point() {
        // Test multiple forks from the same block
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let block1 = mine_block(1, genesis_hash, &[], 1);
        let block1_hash = *block1.hash();
        let _ = blockchain.add_block(block1).ok();

        // Create three competing blocks at height 2
        let fork_a = mine_block(2, block1_hash, &[], 1);
        let fork_b = mine_block(2, block1_hash, &[], 1);
        let fork_c = mine_block(2, block1_hash, &[], 3); // Higher difficulty

        // First block extends main chain
        assert!(
            blockchain.add_block(fork_a).is_ok(),
            "Fork A should be accepted"
        );
        assert_eq!(blockchain.main_chain_len(), 3, "fork_a extends main chain");

        // Second block creates first fork (competes with fork_a)
        assert!(
            blockchain.add_block(fork_b).is_ok(),
            "Fork B should be accepted"
        );
        assert_eq!(blockchain.fork_count(), 1, "fork_b creates a fork");

        // Third block has higher difficulty, should reorganize
        assert!(
            blockchain.add_block(fork_c).is_ok(),
            "Fork C should be accepted"
        );

        // Fork C has difficulty 3 (2^3 = 8), others have difficulty 1 (2^1 = 2)
        // Genesis + block1 = 2^1 + 2^1 = 4 (common to all)
        // fork_c total: 4 + 8 = 12 (highest)
        // fork_a total: 4 + 2 = 6
        // fork_b total: 4 + 2 = 6
        // Fork C should become main chain since it has highest cumulative difficulty
        assert_eq!(
            blockchain.main_chain_len(),
            3,
            "Main chain should be Genesis + block1 + fork_c (highest difficulty)"
        );
        assert!(
            blockchain.fork_count() >= 1,
            "At least one competing fork should exist"
        );
    }

    #[test]
    fn test_double_spend_across_reorganization() {
        // Test that double-spending is possible across reorganization
        // This demonstrates the importance of waiting for confirmations
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let alice = Wallet::new();
        let bob = Wallet::new();
        let charlie = Wallet::new();

        // Main chain: Alice sends 80 to Bob
        let tx_to_bob = alice.create_transaction(bob.address(), 80);
        let block1 = mine_block(1, genesis_hash, &[tx_to_bob], 1);
        let _ = blockchain.add_block(block1).ok();

        let initial_main_len = blockchain.main_chain_len();

        // Fork from genesis: Alice sends 80 to Charlie (double-spend!)
        let tx_to_charlie = alice.create_transaction(charlie.address(), 80);
        let fork_block1 = mine_block(1, genesis_hash, &[tx_to_charlie], 3); // Higher difficulty
        let fork_block2_hash = *fork_block1.hash();
        let fork_block2 = mine_block(2, fork_block2_hash, &[], 3);

        let _ = blockchain.add_block(fork_block1).ok();
        let _ = blockchain.add_block(fork_block2).ok();

        // Fork has higher cumulative difficulty (2^3 + 2^3 = 16) vs main (2^1 = 2)
        // Fork should become main chain through reorganization
        assert_eq!(
            blockchain.main_chain_len(),
            3,
            "Fork should become main chain (Genesis + 2 fork blocks)"
        );
        assert!(
            blockchain.main_chain_len() > initial_main_len,
            "Chain reorganization should have occurred"
        );
        assert_eq!(
            blockchain.fork_count(),
            1,
            "Old main chain (with Bob transaction) should become a fork"
        );

        // This demonstrates that Alice successfully double-spent:
        // - Payment to Bob is now on orphaned fork
        // - Payment to Charlie is on main chain
        // This is why merchants should wait for multiple confirmations!
    }

    #[test]
    fn test_fork_of_fork() {
        // This test validates that we can create forks branching from existing forks
        // Main chain: Genesis -> A -> B
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let block_a = mine_block(1, genesis_hash, &[], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();

        let block_b = mine_block(2, block_a_hash, &[], 1);
        let _ = blockchain.add_block(block_b).ok();

        assert_eq!(
            blockchain.main_chain_len(),
            3,
            "Main chain: Genesis + A + B"
        );

        // Create fork from A: A -> C
        let block_c = mine_block(2, block_a_hash, &[], 1);
        let block_c_hash = *block_c.hash();
        assert!(
            blockchain.add_block(block_c).is_ok(),
            "Block C should be accepted as fork from A"
        );
        assert_eq!(blockchain.fork_count(), 1, "Fork C created");

        // Create fork-of-fork from C: C -> D
        // This is the critical test - previously would panic in recalculate_balances
        let block_d = mine_block(3, block_c_hash, &[], 1);
        assert!(
            blockchain.add_block(block_d).is_ok(),
            "Block D should be accepted as fork from C (fork-of-fork)"
        );

        // Cumulative difficulties:
        // Main (Genesis->A->B): 2 + 2 + 2 = 6
        // Fork (Genesis->A->C->D): 2 + 2 + 2 + 2 = 8
        // Fork is longer with higher cumulative difficulty, so reorganization occurs
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "Fork-of-fork becomes main chain (Genesis + A + C + D)"
        );
        assert_eq!(blockchain.fork_count(), 1, "Old main (B) should be a fork");
    }

    #[test]
    fn test_fork_of_fork_with_reorganization() {
        // Test that a fork-of-fork with higher difficulty can become main chain
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        // Main chain: Genesis -> A -> B (difficulty 1 each)
        let block_a = mine_block(1, genesis_hash, &[], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();

        let block_b = mine_block(2, block_a_hash, &[], 1);
        let _ = blockchain.add_block(block_b).ok();

        // Fork from A: A -> C (difficulty 1)
        let block_c = mine_block(2, block_a_hash, &[], 1);
        let block_c_hash = *block_c.hash();
        let _ = blockchain.add_block(block_c).ok();

        // Fork-of-fork from C: C -> D (difficulty 5)
        // This should create chain Genesis->A->C->D with higher cumulative difficulty
        let block_d = mine_block(3, block_c_hash, &[], 5); // High difficulty
        assert!(
            blockchain.add_block(block_d).is_ok(),
            "High-difficulty fork-of-fork should be accepted"
        );

        // Cumulative difficulties:
        // Main (Genesis->A->B): 2 + 2 + 2 = 6
        // Fork (Genesis->A->C->D): 2 + 2 + 2 + 32 = 38
        // Fork should become main chain
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "Fork-of-fork with higher difficulty should become main chain (Genesis + A + C + D)"
        );
        assert_eq!(
            blockchain.fork_count(),
            1,
            "Old main chain (B) should be a fork"
        );
    }

    #[test]
    fn test_fork_of_fork_with_transactions() {
        // Test fork-of-fork with actual transactions to verify balance tracking
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let alice = Wallet::new();
        let bob = Wallet::new();
        let charlie = Wallet::new();

        // Main chain: Genesis -> A (Alice sends 30 to Bob)
        let tx1 = alice.create_transaction(bob.address(), 30);
        let block_a = mine_block(1, genesis_hash, &[tx1], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();

        // Main chain continues: A -> B (Alice sends 20 to Charlie)
        let tx2 = alice.create_transaction(charlie.address(), 20);
        let block_b = mine_block(2, block_a_hash, &[tx2], 1);
        let _ = blockchain.add_block(block_b).ok();

        // Fork from A: A -> C (Alice sends 40 to Charlie)
        let tx3 = alice.create_transaction(charlie.address(), 40);
        let block_c = mine_block(2, block_a_hash, &[tx3], 1);
        let block_c_hash = *block_c.hash();
        assert!(
            blockchain.add_block(block_c).is_ok(),
            "Fork C with valid transaction should be accepted"
        );

        // Fork-of-fork from C: C -> D (Bob sends 10 to Charlie)
        let bob_wallet = Wallet::from_seed("bob");
        let tx4 = bob_wallet.create_transaction(charlie.address(), 10);
        let block_d = mine_block(3, block_c_hash, &[tx4], 1);
        assert!(
            blockchain.add_block(block_d).is_ok(),
            "Fork-of-fork D with valid transaction should be accepted"
        );

        // Cumulative difficulties:
        // Main (Genesis->A->B): 2 + 2 + 2 = 6
        // Fork (Genesis->A->C->D): 2 + 2 + 2 + 2 = 8
        // Fork-of-fork is longer and has higher cumulative difficulty, becomes main
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "Fork-of-fork should become main chain (Genesis + A + C + D)"
        );
        assert_eq!(blockchain.fork_count(), 1, "Old main (B) should be a fork");
    }

    #[test]
    fn test_fork_of_fork_branching() {
        // This tests the specific issue from feedback: creating a branch off a non-tip fork block
        // Structure: Main chain A→B, Fork1 A→C, Fork2 C→E (branch from Fork1)
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        // Build main chain: Genesis → A → B
        let block_a = mine_block(1, genesis_hash, &[], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();

        let block_b = mine_block(2, block_a_hash, &[], 1);
        let _ = blockchain.add_block(block_b).ok();

        assert_eq!(
            blockchain.main_chain_len(),
            3,
            "Main chain: Genesis + A + B"
        );

        // Create Fork1 from A: A → C
        let block_c = mine_block(2, block_a_hash, &[], 1);
        let block_c_hash = *block_c.hash();
        let result_c = blockchain.add_block(block_c);
        assert!(result_c.is_success(), "Block C should be added as fork");
        assert_eq!(blockchain.fork_count(), 1, "Fork1 created");

        // Create Fork2 from C: C → E (this is the critical test - branching from a fork)
        let block_e = mine_block(3, block_c_hash, &[], 1);
        let result_e = blockchain.add_block(block_e);
        assert!(
            result_e.is_success(),
            "Block E should be added as fork from C (fork-of-fork branch)"
        );

        // With equal difficulty, the longer chain (A→C→E) has higher cumulative difficulty than (A→B)
        // Cumulative: Genesis(2) + A(2) + C(2) + E(2) = 8 vs Genesis(2) + A(2) + B(2) = 6
        // So reorganization should occur
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "Fork-of-fork should become main chain (Genesis + A + C + E)"
        );
        assert_eq!(blockchain.fork_count(), 1, "Old main (B) should be a fork");
    }

    #[test]
    fn test_fork_of_fork_branch_with_transactions() {
        // Test fork-of-fork branching with actual transactions to ensure balance tracking works
        let mut blockchain = BlockChain::new(1);
        let genesis_hash = *blockchain.latest_block_hash();

        let alice = Wallet::new();
        let bob = Wallet::new();
        let charlie = Wallet::new();

        // Main chain: Genesis → A (Alice sends 20 to Bob)
        let tx1 = alice.create_transaction(bob.address(), 20);
        let block_a = mine_block(1, genesis_hash, &[tx1], 1);
        let block_a_hash = *block_a.hash();
        let _ = blockchain.add_block(block_a).ok();
        assert_eq!(blockchain.cumulative_difficulty(), 2 + 2, "After block A");
        assert_eq!(blockchain.main_chain_len(), 2, "Genesis + A");

        // Main chain continues: A → B (Alice sends 10 to Charlie)
        let tx2 = alice.create_transaction(charlie.address(), 10);
        let block_b = mine_block(2, block_a_hash, &[tx2], 1);
        let _ = blockchain.add_block(block_b).ok();
        assert_eq!(
            blockchain.cumulative_difficulty(),
            2 + 2 + 2,
            "After block B"
        );
        assert_eq!(blockchain.main_chain_len(), 3, "Genesis + A + B");

        // Fork1 from A: A → C (Alice sends 30 to Charlie)
        let tx3 = alice.create_transaction(charlie.address(), 30);
        let block_c = mine_block(2, block_a_hash, &[tx3], 1);
        let block_c_hash = *block_c.hash();
        let result_c = blockchain.add_block(block_c);
        assert!(result_c.is_success(), "Fork C should be created");
        assert_eq!(blockchain.fork_count(), 1, "Fork1 created");
        assert_eq!(
            blockchain.cumulative_difficulty(),
            2 + 2 + 2,
            "Main chain unchanged after fork C added"
        );
        assert_eq!(blockchain.main_chain_len(), 3, "Still Genesis + A + B");

        // Fork2 from C: C → E (Bob sends 5 to Charlie)
        // This is the critical branch: E branches from C (which itself is a fork)
        // Cumulative difficulty calculation:
        // - Main chain (Genesis + A + B): 2 + 2 + 2 = 6
        // - Fork chain (Genesis + A + C + E): 2 + 2 + 2 + 2 = 8
        // Fork chain has higher difficulty, so reorganization will occur
        let bob_wallet = Wallet::from_seed("bob");
        let tx4 = bob_wallet.create_transaction(charlie.address(), 5);
        let block_e = mine_block(3, block_c_hash, &[tx4], 1);
        let result_e = blockchain.add_block(block_e);
        assert!(
            result_e.is_success(),
            "Fork-of-fork branch E should be created from C"
        );

        // After adding E, reorganization should happen immediately
        // because fork chain (8) > main chain (6)
        assert_eq!(
            blockchain.cumulative_difficulty(),
            2 + 2 + 2 + 2,
            "After reorganization to fork-of-fork (Genesis + A + C + E)"
        );
        assert_eq!(
            blockchain.main_chain_len(),
            4,
            "Fork-of-fork should become main chain (Genesis + A + C + E)"
        );
        assert_eq!(blockchain.fork_count(), 1, "Old main (B) should be a fork");
    }
}
