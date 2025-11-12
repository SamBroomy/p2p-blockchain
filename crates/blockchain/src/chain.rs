use std::{collections::HashMap, fmt::Debug, marker::PhantomData};

use blake3::Hash;
use blockchain_types::{
    Block, Transaction,
    consts::{GENESIS_ROOT_HASH, WALLET_INITIAL_BALANCE},
    wallet::Address,
};
use serde::{Deserialize, Serialize};

type Balance = u64;
type Delta = i64;
// As our implementation dosent have block rewards yet, everyone starts with an initial balance to make things easier
pub const INITIAL_BALANCE: Balance = WALLET_INITIAL_BALANCE;
const INITIAL_DELTA: Delta = INITIAL_BALANCE as Delta;
type AccountBalances = HashMap<Address, Balance>;
type AccountDelta = HashMap<Address, Delta>;

pub trait ChainType: sealed::Sealed {}

/// A root chain - connects to genesis, balances must be non-negative
#[derive(Debug)]
pub struct RootChain;
/// A fork chain - may not connect to genesis, must connect to a root chain at some point
/// Cant fully validate transactions without root chain. But can validate transactions within the fork itself
#[derive(Debug)]
pub struct ForkChain;

impl ChainType for RootChain {}
impl ChainType for ForkChain {}
// Seal the trait so users can't implement their own chain types
mod sealed {
    pub trait Sealed {}
    impl Sealed for super::RootChain {}
    impl Sealed for super::ForkChain {}
}

#[inline]
// can the tips touch? lol
fn touch_tips(tip: &Hash, root: &Hash) -> bool {
    tip == root
}

#[inline]
pub fn is_connecting_block_valid(previous_block: &Block, new_block: &Block) -> bool {
    touch_tips(previous_block.hash(), new_block.previous_hash())
}
#[inline]
fn update_deltas(deltas: &mut AccountDelta, txs: &[Transaction]) {
    for tx in txs {
        *deltas.entry(*tx.receiver()).or_insert(0) += tx.amount() as Delta;
        if let Some(sender) = tx.sender() {
            *deltas.entry(*sender).or_insert(0) -= tx.amount() as Delta;
        }
    }
}
fn get_transaction_deltas(txs: &[Transaction]) -> AccountDelta {
    let mut deltas = HashMap::new();
    update_deltas(&mut deltas, txs);
    deltas
}
fn validate_balance_deltas(balances: &AccountBalances, deltas: &AccountDelta) -> bool {
    deltas.iter().all(|(address, delta)| {
        let current_balance = *balances.get(address).unwrap_or(&INITIAL_BALANCE) as Delta;
        current_balance + *delta >= 0
    })
}
fn apply_deltas_to_balance_deltas(deltas: &mut AccountDelta, new_deltas: AccountDelta) {
    for (key, delta) in new_deltas {
        let entry = deltas.entry(key).or_insert(0);
        *entry += delta;
    }
}

/// Assumes deltas are valid (i.e. no negative balances will result)
fn apply_deltas_to_balances(balances: &mut AccountBalances, deltas: AccountDelta) {
    for (key, delta) in deltas {
        let entry = balances.entry(key).or_insert(INITIAL_BALANCE);
        // Safe to unwrap as we validated before
        let current = *entry as Delta;
        let new_balance = current + delta;
        // ⚡ Debug assertion to catch bugs
        debug_assert!(
            new_balance >= 0,
            "Balance would go negative: {current} + {delta} = {new_balance} for address {key:?}",
        );
        *entry = new_balance as Balance;
    }
}

/// Recalculates balances from genesis (first block must connect to `GENESIS_ROOT_HASH`)
pub fn recalculate_balances(blocks: &[Block]) -> HashMap<Address, Balance> {
    assert!(
        !blocks.is_empty(),
        "Cannot recalculate balances of an empty chain"
    );
    assert_eq!(
        blocks
            .first()
            .expect("there is at least one block to recalculate balances")
            .previous_hash(),
        &GENESIS_ROOT_HASH,
        "Can only recalculate balances from genesis"
    );
    recalculate_balances_from(HashMap::new(), blocks)
}

/// Recalculates balances from a starting balance state (for fork chains)
fn recalculate_balances_from(mut balances: AccountBalances, blocks: &[Block]) -> AccountBalances {
    for block in blocks {
        let deltas = get_transaction_deltas(block.transactions());
        debug_assert!(
            validate_balance_deltas(&balances, &deltas),
            "Invalid balances in chain during recalculation"
        );
        apply_deltas_to_balances(&mut balances, deltas);
    }
    balances
}

/// Helper: Reverse transactions from blocks[`start_idx`..] to "undo" their effects
/// Used to calculate balances at earlier points in a chain
fn reverse_transactions_from(
    starting_balances: &AccountBalances,
    blocks: &[Block],
    start_idx: usize,
) -> AccountBalances {
    debug_assert!(
        start_idx <= blocks.len(),
        "Start index {start_idx} must be <= blocks length {}",
        blocks.len()
    );
    let mut balances = starting_balances.clone();

    // Process blocks in reverse order from the end back to start_idx
    for block in blocks[start_idx..].iter().rev() {
        // Reverse transactions within each block
        for tx in block.transactions().iter().rev() {
            // Undo: receiver += amount, sender -= amount
            let receiver_balance = balances.entry(*tx.receiver()).or_insert(INITIAL_BALANCE);
            *receiver_balance -= tx.amount();

            if let Some(sender) = tx.sender() {
                let sender_balance = balances.entry(*sender).or_insert(INITIAL_BALANCE);
                *sender_balance += tx.amount();
            }
        }
    }

    balances
}
fn verify_balances(_balances: &AccountBalances) -> bool {
    // Balance is u64 (unsigned), so it can never be negative
    // Actual balance validation happens in validate_balance_deltas before applying changes
    // This function is kept for potential future invariant checks
    true
}
// A valid chain must have at least one block
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound = "")]
pub struct Chain<T: ChainType = RootChain> {
    /// Ordered sequence of blocks
    blocks: Vec<Block>,
    /// Must be valid non-negative balances.
    balances: AccountBalances,
    #[serde(skip)]
    _marker: PhantomData<T>,
}
impl<T: ChainType> Chain<T> {
    pub fn tip_block(&self) -> &Block {
        self.blocks.last().expect("Chain has at least one block")
    }

    pub fn tip_hash(&self) -> &Hash {
        self.tip_block().hash()
    }

    pub(super) fn root_block(&self) -> &Block {
        self.blocks.first().expect("Chain has at least one block")
    }

    pub fn root_hash(&self) -> &Hash {
        self.root_block().hash()
    }

    pub fn get_balance(&self, address: &Address) -> Option<Balance> {
        self.balances.get(address).copied()
    }

    pub fn cumulative_difficulty(&self) -> u128 {
        self.blocks
            .iter()
            .map(|block| {
                // work = 2^256 / target
                // for leading zeros difficulty: work ≈ 2^(difficulty)
                let difficulty = block.difficulty() as u32;
                // Cap at 127 to prevent overflow with u128
                if difficulty > 127 {
                    u128::MAX
                } else {
                    1u128 << difficulty // This is 2^difficulty
                }
            })
            .sum()
    }

    // Check the chain has a proper block linage
    fn is_structurally_valid(&self) -> bool {
        debug_assert!(
            !self.blocks.is_empty(),
            "Chain should have at least one block for structural validity check"
        );
        // To check the first block is valid and then all other blocks are checked in the loop
        // We kinda don't need this as blocks should be valid but its worth it for now.
        self.root_block().is_valid()
            && self.blocks.windows(2).all(|w| {
                let previous_block = &w[0];
                let current_block = &w[1];
                previous_block.index() + 1 == current_block.index()
                    && current_block.is_valid()
                    && is_connecting_block_valid(previous_block, current_block)
            })
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub(super) fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Find the index of a block with the given hash
    pub fn find_block_index(&self, hash: &Hash) -> Option<usize> {
        self.blocks.iter().position(|b| b.hash() == hash)
    }

    /// Check if this chain contains a block with the given hash
    pub fn contains_block(&self, hash: &Hash) -> bool {
        self.find_block_index(hash).is_some()
    }

    pub fn add_block(&mut self, block: Block) -> Result<(), Block> {
        if !block.is_valid() || !is_connecting_block_valid(self.tip_block(), &block) {
            return Err(block);
        }
        let deltas = get_transaction_deltas(block.transactions());
        if !validate_balance_deltas(&self.balances, &deltas) {
            return Err(block);
        }
        apply_deltas_to_balances(&mut self.balances, deltas);
        debug_assert_eq!(
            self.tip_block().index() + 1,
            block.index(),
            "Block must be sequential"
        );
        self.blocks.push(block);
        Ok(())
    }

    pub fn validate_transaction(&self, tx: &Transaction) -> bool {
        let sender_balance = tx
            .sender()
            .and_then(|addr| self.get_balance(addr))
            .unwrap_or(INITIAL_BALANCE);
        sender_balance >= tx.amount()
    }
}

impl Chain<RootChain> {
    pub fn new_from_genesis(genesis: Block) -> Self {
        assert_eq!(
            genesis.previous_hash(),
            &GENESIS_ROOT_HASH,
            "Genesis block must connect to GENESIS_ROOT_HASH"
        );
        assert!(genesis.is_valid(), "Genesis block is invalid");
        let mut balances = HashMap::new();
        let deltas = get_transaction_deltas(genesis.transactions());
        assert!(
            validate_balance_deltas(&balances, &deltas),
            "Invalid genesis block balances"
        );
        apply_deltas_to_balances(&mut balances, deltas);
        Self {
            blocks: vec![genesis],
            balances,
            _marker: PhantomData,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.is_structurally_valid()
            && (self.root_block().previous_hash() == &GENESIS_ROOT_HASH)
            && self.verify_balances()
    }

    pub fn truncate_at(&mut self, hash: &Hash) -> Option<Chain<ForkChain>> {
        let index = self.find_block_index(hash)?;
        let split_blocks = self.blocks.split_off(index + 1);
        if split_blocks.is_empty() {
            // No blocks split, self.blocks remains unchanged
            return None;
        }
        // Recalculate balances for the truncated root chain (up to and including the split point)
        let updated_root_balances = recalculate_balances(&self.blocks);
        // The fork gets the OLD balances (which include the split_blocks' transactions)
        // This is correct because the fork contains those blocks
        // We swap in the recalculated balances for the now-shorter root chain
        let fork_balances = std::mem::replace(&mut self.balances, updated_root_balances);

        assert!(
            self.is_valid(),
            "Root chain is invalid after truncation at {hash:?}"
        );

        Some(Chain {
            blocks: split_blocks,
            balances: fork_balances,
            _marker: PhantomData,
        })
    }

    fn verify_balances(&self) -> bool {
        let recalculated_balances = recalculate_balances(&self.blocks);
        self.balances == recalculated_balances && verify_balances(&self.balances)
    }

    pub fn cumulative_difficulty_till(&self, index: usize) -> u128 {
        self.blocks
            .iter()
            // Take blocks up to and including index
            .take(index + 1)
            .map(|block| {
                let difficulty = block.difficulty() as u32;
                if difficulty > 127 {
                    u128::MAX
                } else {
                    1u128 << difficulty
                }
            })
            .sum()
    }

    /// Assumes the fork chain is valid against this root chain
    pub fn append_fork(&mut self, fork_chain: Chain<ForkChain>) {
        assert!(
            fork_chain.valid_fork(self),
            "Fork chain is not valid against root chain"
        );
        // Could loop through and apply each block, but we know the fork is valid so just append directly
        // for block in fork_chain.blocks {
        //     let deltas = get_transaction_deltas(block.transactions());
        //     assert!(
        //         validate_balance_deltas(&self.balances, &deltas),
        //         "Invalid balances when appending fork"
        //     );
        //     apply_deltas_to_balances(&mut self.balances, deltas);
        //     self.blocks.push(block);
        // }
        self.blocks.extend(fork_chain.blocks);
        self.balances = fork_chain.balances;
        assert!(
            self.is_valid(),
            "Root chain is invalid after appending fork chain"
        );
    }

    // What we are doing is basically finding where the fork connects and seeing if the new cumulative difficulty is higher than the current chain then truncating at that point and appending the fork
    pub fn merge_fork(
        &mut self,
        fork_chain: Chain<ForkChain>,
    ) -> Result<Chain<ForkChain>, Chain<ForkChain>> {
        if !fork_chain.valid_fork(self) {
            return Err(fork_chain);
        }
        // Find the fork connection point
        let connection_hash = fork_chain.root_block().previous_hash();

        // Handle fork from genesis specially
        let (connection_idx, root_cumulative_difficulty) = if *connection_hash == GENESIS_ROOT_HASH
        {
            // Fork from genesis - no blocks before it
            (None, 0u128)
        } else {
            let Some(idx) = self.find_block_index(connection_hash) else {
                return Err(fork_chain);
            };
            (Some(idx), self.cumulative_difficulty_till(idx))
        };

        let chain_cumulative_difficulty = self.cumulative_difficulty();
        let fork_cumulative_difficulty = fork_chain.cumulative_difficulty();
        if root_cumulative_difficulty + fork_cumulative_difficulty <= chain_cumulative_difficulty {
            return Err(fork_chain);
        }

        // Truncate at the connection point and append fork
        let old_fork: Chain<ForkChain> = if connection_idx.is_some() {
            // Normal fork - truncate at connection point
            let Some(old_fork) = self.truncate_at(connection_hash) else {
                return Err(fork_chain);
            };
            old_fork
        } else {
            // Save ALL current blocks (including genesis) as the old fork
            let old_blocks = std::mem::take(&mut self.blocks);
            let old_balances = std::mem::take(&mut self.balances);
            // Replace entire chain with fork
            self.blocks = fork_chain.blocks;
            self.balances = fork_chain.balances;
            assert!(
                self.is_valid(),
                "Root chain must be valid after merging genesis fork"
            );
            // Return ALL old blocks (including the old genesis)
            return Ok(Chain {
                blocks: old_blocks,
                balances: old_balances,
                _marker: PhantomData,
            });
        };
        // Swap in the fork chain (for normal forks)
        self.append_fork(fork_chain);
        // Return the old main chain tail
        Ok(old_fork)
    }
}

impl Chain<ForkChain> {
    /// Create a new fork from a single block with given starting balances
    pub(super) fn new_from_block(
        block: Block,
        starting_balances: AccountBalances,
    ) -> Result<Self, Block> {
        if !block.is_valid() {
            return Err(block);
        }

        let mut balances = starting_balances;
        let deltas = get_transaction_deltas(block.transactions());
        if !validate_balance_deltas(&balances, &deltas) {
            return Err(block);
        }
        apply_deltas_to_balances(&mut balances, deltas);
        debug_assert!(
            verify_balances(&balances),
            "Fork balances must be valid after creation"
        );
        Ok(Self {
            blocks: vec![block],
            balances,
            _marker: PhantomData,
        })
    }

    /// Get balances at a specific block in this fork chain (public interface)
    /// Used for creating sub-forks from a fork
    pub(super) fn balances_at(&self, hash: &Hash) -> Option<AccountBalances> {
        let index = self.find_block_index(hash)?;
        debug_assert!(
            index < self.blocks.len(),
            "Index {index} out of bounds for chain of length {}",
            self.blocks.len()
        );
        Some(self.balances_at_index(index))
    }

    /// Core implementation: Get balances at a specific index (avoids redundant hash lookups)
    fn balances_at_index(&self, index: usize) -> AccountBalances {
        // We need to work backwards from current balances to find balances at the given block
        // Fork current balances = starting_balances + transactions[0..=last_index]
        // We want: starting_balances + transactions[0..=index]
        debug_assert!(
            index < self.blocks.len(),
            "Index {index} out of bounds for chain of length {}",
            self.blocks.len()
        );
        // Reverse transactions from blocks after the target block to "undo" them
        reverse_transactions_from(&self.balances, &self.blocks, index + 1)
    }

    pub fn truncate_at(&mut self, hash: &Hash) -> Option<Self> {
        let index = self.find_block_index(hash)?;

        // Calculate balances BEFORE split_off, while we can still see all blocks
        let truncated_balances = self.balances_at_index(index);

        let split_blocks = self.blocks.split_off(index + 1);
        if split_blocks.is_empty() {
            // No blocks split, self.blocks remains unchanged
            return None;
        }

        // Calculate fork balances from the truncation point
        let fork_balances = recalculate_balances_from(truncated_balances.clone(), &split_blocks);
        // Update self with truncated state
        self.balances = truncated_balances;
        debug_assert!(
            !self.blocks.is_empty(),
            "Fork chain cannot be empty after truncation"
        );
        debug_assert!(
            !split_blocks.is_empty(),
            "Split blocks cannot be empty if we're returning Some"
        );

        Some(Self {
            blocks: split_blocks,
            balances: fork_balances,
            _marker: PhantomData,
        })
    }

    pub fn is_valid_against_root(&self, hash: &Hash, root_balances: &AccountBalances) -> bool {
        if !touch_tips(hash, self.root_hash()) {
            return false;
        }

        // Root account balances + (blocks tx in fork delta) == fork account balances
        let mut balances = root_balances.clone();
        for block in &self.blocks {
            let deltas = get_transaction_deltas(block.transactions());
            if !validate_balance_deltas(&balances, &deltas) {
                return false;
            }
            apply_deltas_to_balances(&mut balances, deltas);
        }
        self.is_structurally_valid()
            && verify_balances(&self.balances)
            && (self.balances == balances)
    }

    pub fn valid_fork(&self, root_chain: &Chain<RootChain>) -> bool {
        // The fork connects to the root chain at fork.root_block().previous_hash()
        // This hash should be in the root chain
        let connection_hash = self.root_block().previous_hash();

        // Check if the connection point exists in root chain
        let Some(connection_idx) = root_chain.find_block_index(connection_hash) else {
            // Connection point not in root chain, check if it's genesis
            if *connection_hash != GENESIS_ROOT_HASH {
                return false;
            }
            debug_assert!(
                verify_balances(&self.balances),
                "Fork from genesis must have valid (non-negative) balances"
            );
            debug_assert!(
                {
                    let mut expected_balances = HashMap::new();
                    for block in &self.blocks {
                        let deltas = get_transaction_deltas(block.transactions());
                        if !validate_balance_deltas(&expected_balances, &deltas) {
                            return false;
                        }
                        apply_deltas_to_balances(&mut expected_balances, deltas);
                    }

                    self.balances == expected_balances
                },
                "Fork from genesis must have balances matching its transactions"
            );
            // Fork connects to genesis, use empty balances from genesis
            return self.is_structurally_valid() && verify_balances(&self.balances);
        };

        // Get balances at the connection point (after including the connection block)
        let connection_balances = recalculate_balances(&root_chain.blocks[..=connection_idx]);

        // Validate the fork's structure and that its balances match what we'd get
        // by applying its transactions starting from the connection point balances
        let mut balances = connection_balances;
        for block in &self.blocks {
            let deltas = get_transaction_deltas(block.transactions());
            if !validate_balance_deltas(&balances, &deltas) {
                return false;
            }
            apply_deltas_to_balances(&mut balances, deltas);
        }
        self.is_structurally_valid()
            && verify_balances(&self.balances)
            && (self.balances == balances)
    }

    /// Assumes the fork chain is valid against this root chain
    pub fn append_fork(&mut self, fork_chain: Self) {
        debug_assert!(
            is_connecting_block_valid(self.tip_block(), fork_chain.root_block()),
            "Fork must connect to this chain's tip"
        );
        self.blocks.extend(fork_chain.blocks);
        self.balances = fork_chain.balances;
        debug_assert!(
            self.is_structurally_valid(),
            "Chain must be structurally valid after append"
        );
    }
}

#[cfg(test)]
mod tests {
    use blockchain_types::{BlockConstructor, MinerSimple, MiningStrategy, wallet::Wallet};

    use super::*;

    fn mine_block(
        index: u64,
        previous_hash: Hash,
        transactions: &[Transaction],
        difficulty: usize,
    ) -> Block {
        MinerSimple::mine(
            BlockConstructor::new(index, transactions, previous_hash, None),
            difficulty,
            None,
        )
    }

    #[test]
    fn test_root_chain_creation() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let chain = Chain::<RootChain>::new_from_genesis(genesis);

        assert_eq!(chain.len(), 1);
        assert!(chain.is_valid());
    }

    #[test]
    fn test_add_block_to_root_chain() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 1);
        assert!(chain.add_block(block1).is_ok());
        assert_eq!(chain.len(), 2);
        assert!(chain.is_valid());
    }

    #[test]
    fn test_truncate_at_preserves_balances() {
        // This is the critical test for the truncate_at logic
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        // Create transactions with known balance changes
        let alice = Wallet::from_seed("alice");
        let bob = Wallet::from_seed("bob");
        let charlie = Wallet::from_seed("charlie");

        // Block 1: Alice sends 30 to Bob
        let tx1 = alice.create_transaction(bob.address(), 30);
        let block1 = mine_block(1, *genesis.hash(), &[tx1], 1);
        let block1_hash = *block1.hash();
        chain.add_block(block1).unwrap();

        // Block 2: Alice sends 20 to Charlie
        let tx2 = alice.create_transaction(charlie.address(), 20);
        let block2 = mine_block(2, block1_hash, &[tx2], 1);
        let block2_hash = *block2.hash();
        chain.add_block(block2.clone()).unwrap();

        // Block 3: Bob sends 10 to Charlie
        let bob_wallet = Wallet::from_seed("bob");
        let tx3 = bob_wallet.create_transaction(charlie.address(), 10);
        let block3 = mine_block(3, block2_hash, &[tx3], 1);
        chain.add_block(block3.clone()).unwrap();

        // At this point:
        // Alice: 100 - 30 - 20 = 50
        // Bob: 100 + 30 - 10 = 120
        // Charlie: 100 + 20 + 10 = 130

        assert_eq!(chain.len(), 4); // Genesis + 3 blocks

        // Now truncate at block1 (split off block2 and block3)
        let fork = chain.truncate_at(&block1_hash).unwrap();

        // Verify root chain state after truncate
        assert_eq!(chain.len(), 2); // Genesis + block1
        assert!(chain.is_valid());

        // Root chain balances should be after block1:
        // Alice: 100 - 30 = 70
        // Bob: 100 + 30 = 130
        // Charlie: 100
        let alice_balance_root = chain.get_balance(alice.address()).unwrap_or(100);
        let bob_balance_root = chain.get_balance(bob.address()).unwrap_or(100);
        let charlie_balance_root = chain.get_balance(charlie.address()).unwrap_or(100);

        assert_eq!(
            alice_balance_root, 70,
            "Alice should have 70 after block1 in root chain"
        );
        assert_eq!(
            bob_balance_root, 130,
            "Bob should have 130 after block1 in root chain"
        );
        assert_eq!(
            charlie_balance_root, 100,
            "Charlie should have 100 (no transactions yet) in root chain"
        );

        // Verify fork state
        assert_eq!(fork.len(), 2); // block2 + block3

        // Fork balances should be after block3 (the original tip):
        // Alice: 100 - 30 - 20 = 50
        // Bob: 100 + 30 - 10 = 120
        // Charlie: 100 + 20 + 10 = 130
        let alice_balance_fork = fork.get_balance(alice.address()).unwrap_or(100);
        let bob_balance_fork = fork.get_balance(bob.address()).unwrap_or(100);
        let charlie_balance_fork = fork.get_balance(charlie.address()).unwrap_or(100);

        assert_eq!(
            alice_balance_fork, 50,
            "Fork: Alice should have 50 after all transactions"
        );
        assert_eq!(
            bob_balance_fork, 120,
            "Fork: Bob should have 120 after all transactions"
        );
        assert_eq!(
            charlie_balance_fork, 130,
            "Fork: Charlie should have 130 after all transactions"
        );

        // Verify fork is valid against the root chain
        assert!(
            fork.valid_fork(&chain),
            "Fork should be valid against root chain"
        );
    }

    #[test]
    fn test_truncate_at_tip_returns_none() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 1);
        chain.add_block(block1.clone()).unwrap();

        // Truncating at the tip should return None (no blocks to split off)
        let result = chain.truncate_at(block1.hash());
        assert!(result.is_none());
        assert_eq!(chain.len(), 2); // Chain unchanged
    }

    #[test]
    fn test_truncate_at_genesis() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 1);
        let block1_hash = *block1.hash();
        chain.add_block(block1).unwrap();

        let block2 = mine_block(2, block1_hash, &[], 1);
        chain.add_block(block2).unwrap();

        // Truncate at genesis (split off everything except genesis)
        let fork = chain.truncate_at(genesis.hash()).unwrap();

        assert_eq!(chain.len(), 1); // Just genesis
        assert_eq!(fork.len(), 2); // block1 + block2
    }

    #[test]
    fn test_fork_chain_balance_validation() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let alice = Wallet::from_seed("alice");
        let bob = Wallet::from_seed("bob");

        // Block 1: Alice sends 30 to Bob (valid)
        let tx1 = alice.create_transaction(bob.address(), 30);
        let block1 = mine_block(1, *genesis.hash(), &[tx1], 1);
        chain.add_block(block1).unwrap();

        // Create a fork from genesis with a transaction that would overdraw
        let tx_invalid = alice.create_transaction(bob.address(), 150); // Alice only has 100!
        let invalid_block = mine_block(1, GENESIS_ROOT_HASH, &[tx_invalid], 1);

        // Creating a fork with invalid balances should fail
        let result = Chain::<ForkChain>::new_from_block(invalid_block, HashMap::new());
        assert!(
            result.is_err(),
            "Fork with negative balance should be rejected"
        );
    }

    #[test]
    fn test_fork_valid_fork_checks_connection() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut root_chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 1);
        let block1_hash = *block1.hash();
        root_chain.add_block(block1).unwrap();

        let block2 = mine_block(2, block1_hash, &[], 1);
        root_chain.add_block(block2).unwrap();

        // Create a valid fork from block1
        let fork_block = mine_block(2, block1_hash, &[], 2); // Competing with block2
        let fork_balances = recalculate_balances(&root_chain.blocks()[..=1]); // Balances after block1
        let fork = Chain::<ForkChain>::new_from_block(fork_block, fork_balances).unwrap();

        assert!(fork.valid_fork(&root_chain), "Fork should be valid");

        // Create a fork that doesn't connect (uses a random hash)
        let random_hash = Hash::from_bytes([42u8; 32]);
        let orphan_block = mine_block(99, random_hash, &[], 1);
        let orphan_fork = Chain::<ForkChain>::new_from_block(orphan_block, HashMap::new()).unwrap();

        assert!(
            !orphan_fork.valid_fork(&root_chain),
            "Orphan fork should not be valid"
        );
    }

    #[test]
    fn test_cumulative_difficulty_calculation() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1); // diff 1 = 2^1 = 2
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 2); // diff 2 = 2^2 = 4
        chain.add_block(block1.clone()).unwrap();

        let block2 = mine_block(2, *block1.hash(), &[], 3); // diff 3 = 2^3 = 8
        chain.add_block(block2).unwrap();

        // Cumulative difficulty = 2 + 4 + 8 = 14
        let cumulative = chain.cumulative_difficulty();
        assert_eq!(cumulative, 14);

        // Test cumulative_difficulty_till
        let difficulty_till_1 = chain.cumulative_difficulty_till(1); // genesis + block1 = 2 + 4 = 6
        assert_eq!(difficulty_till_1, 6);
    }

    #[test]
    fn test_append_fork_to_root() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let block1 = mine_block(1, *genesis.hash(), &[], 1);
        let block1_hash = *block1.hash();
        chain.add_block(block1).unwrap();

        // Create a fork
        let fork_balances = recalculate_balances(&chain.blocks()[..=1]);
        let fork_block1 = mine_block(2, block1_hash, &[], 2);
        let fork_block1_hash = *fork_block1.hash();
        let mut fork = Chain::<ForkChain>::new_from_block(fork_block1, fork_balances).unwrap();

        let fork_block2 = mine_block(3, fork_block1_hash, &[], 2);
        fork.add_block(fork_block2).unwrap();

        assert_eq!(fork.len(), 2);

        // Append fork to root chain
        let original_len = chain.len();
        chain.append_fork(fork);

        assert_eq!(chain.len(), original_len + 2);
        assert!(chain.is_valid());
    }

    #[test]
    fn test_fork_from_genesis_is_valid() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut root = Chain::<RootChain>::new_from_genesis(genesis.clone());

        //  Root: Genesis -> A
        let block_a = mine_block(1, *genesis.hash(), &[], 1);
        root.add_block(block_a).unwrap();

        // Fork from genesis: B
        let block_b = mine_block(1, GENESIS_ROOT_HASH, &[], 3);
        let fork = Chain::<ForkChain>::new_from_block(block_b, HashMap::new()).unwrap();

        // Fork should be valid against root
        assert!(
            fork.valid_fork(&root),
            "Single-block fork from genesis should be valid"
        );

        // Fork from genesis: B -> C
        let block_b2 = mine_block(1, GENESIS_ROOT_HASH, &[], 3);
        let block_b2_hash = *block_b2.hash();
        let mut fork2 = Chain::<ForkChain>::new_from_block(block_b2, HashMap::new()).unwrap();
        let block_c = mine_block(2, block_b2_hash, &[], 3);
        fork2.add_block(block_c).unwrap();

        // Two-block fork should also be valid
        assert!(
            fork2.valid_fork(&root),
            "Two-block fork from genesis should be valid"
        );
    }

    #[test]
    fn test_merge_fork_swaps_chains() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut root = Chain::<RootChain>::new_from_genesis(genesis.clone());

        // Main chain: Genesis -> A (low difficulty)
        let block_a = mine_block(1, *genesis.hash(), &[], 1);
        root.add_block(block_a).unwrap();

        // Fork: Genesis -> B -> C (higher difficulty)
        let fork_balances = HashMap::new(); // Fork from genesis
        let block_b = mine_block(1, GENESIS_ROOT_HASH, &[], 3);
        let block_b_hash = *block_b.hash();
        let mut fork = Chain::<ForkChain>::new_from_block(block_b, fork_balances).unwrap();

        let block_c = mine_block(2, block_b_hash, &[], 3);
        fork.add_block(block_c).unwrap();

        // Fork difficulty: 2^3 + 2^3 = 8 + 8 = 16
        // Main difficulty: 2^1 + 2^1 = 2 + 2 = 4
        assert!(fork.cumulative_difficulty() > root.cumulative_difficulty());
        assert!(
            fork.valid_fork(&root),
            "Fork should be valid before merging"
        );

        // Merge fork into root
        let old_main = root.merge_fork(fork).unwrap();

        // Root is COMPLETELY REPLACED by the fork (B + C)
        // B is now the genesis block, followed by C
        assert_eq!(
            root.len(),
            2,
            "Root should have B + C (fork completely replaces old chain)"
        );
        // Old main chain (genesis + A) is returned as a fork
        assert_eq!(
            old_main.len(),
            2,
            "Old main should contain both old genesis and A"
        );
        // Verify the root chain is valid
        assert!(root.is_valid(), "New root chain should be valid");
    }

    #[test]
    fn test_balance_recalculation_from_genesis() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);

        let alice = Wallet::from_seed("alice");
        let bob = Wallet::from_seed("bob");

        let tx1 = alice.create_transaction(bob.address(), 25);
        let block1 = mine_block(1, *genesis.hash(), &[tx1], 1);

        let tx2 = alice.create_transaction(bob.address(), 25);
        let block2 = mine_block(2, *block1.hash(), &[tx2], 1);

        let blocks = vec![genesis, block1, block2];
        let balances = recalculate_balances(&blocks);

        // Alice: 100 - 25 - 25 = 50
        // Bob: 100 + 25 + 25 = 150
        assert_eq!(balances.get(alice.address()), Some(&50));
        assert_eq!(balances.get(bob.address()), Some(&150));
    }

    #[test]
    #[should_panic(expected = "Can only recalculate balances from genesis")]
    fn test_recalculate_balances_requires_genesis() {
        // This should panic because block doesn't start from genesis
        let random_hash = Hash::from_bytes([1u8; 32]);
        let block = mine_block(5, random_hash, &[], 1);
        recalculate_balances(&[block]);
    }

    #[test]
    fn test_genesis_fork_with_transactions_is_valid() {
        // This test proves that genesis forks can have balances != INITIAL_BALANCE
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let mut root = Chain::<RootChain>::new_from_genesis(genesis.clone());

        // Main chain: Genesis -> A
        let block_a = mine_block(1, *genesis.hash(), &[], 1);
        root.add_block(block_a).unwrap();

        // Create a fork from GENESIS_ROOT_HASH with transactions
        // This fork has a different genesis than root's genesis
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Fork block B transfers 50 from wallet1 to wallet2
        let tx = wallet1.create_transaction(wallet2.address(), 50);
        let fork_block = mine_block(1, GENESIS_ROOT_HASH, &[tx], 2);

        // Create fork with empty starting balances (genesis fork)
        let fork = Chain::<ForkChain>::new_from_block(fork_block, HashMap::new()).unwrap();

        // After the transaction, balances should be:
        // wallet1: 100 - 50 = 50
        // wallet2: 100 + 50 = 150
        // NOT all accounts at INITIAL_BALANCE (100)!

        assert_eq!(fork.balances.get(wallet1.address()), Some(&50));
        assert_eq!(fork.balances.get(wallet2.address()), Some(&150));

        // This fork is VALID even though balances != INITIAL_BALANCE
        assert!(
            fork.valid_fork(&root),
            "Fork with transactions should be valid even though balances diverge from INITIAL_BALANCE"
        );
    }

    #[test]
    fn test_validate_transaction() {
        let genesis = mine_block(0, GENESIS_ROOT_HASH, &[], 1);
        let chain = Chain::<RootChain>::new_from_genesis(genesis.clone());

        let alice = Wallet::from_seed("alice");
        let bob = Wallet::from_seed("bob");

        // Valid transaction: Alice sends 50 to Bob
        let tx_valid = alice.create_transaction(bob.address(), 50);
        assert!(
            chain.validate_transaction(&tx_valid),
            "Valid transaction should be accepted"
        );

        // Invalid transaction: Alice tries to send 150 to Bob (only has 100)
        let tx_invalid = alice.create_transaction(bob.address(), 150);
        assert!(
            !chain.validate_transaction(&tx_invalid),
            "Invalid transaction should be rejected"
        );
    }
}
