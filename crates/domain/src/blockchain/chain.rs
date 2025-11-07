use std::{collections::HashMap, fmt::Debug, marker::PhantomData};

use blake3::Hash;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};

use crate::{Transaction, block::Block, blockchain::GENESIS_ROOT_HASH};

type Balance = u64;
type Delta = i64;
// As our implementation dosent have block rewards yet, everyone starts with an initial balance to make things easier
const INITIAL_BALANCE: Balance = 100;
const INITIAL_DELTA: Delta = INITIAL_BALANCE as Delta;
type AccountBalances = HashMap<VerifyingKey, Balance>;
type AccountDelta = HashMap<VerifyingKey, Delta>;

pub trait ChainType: sealed::Sealed {}

/// A root chain - connects to genesis, balances must be non-negative
pub struct RootChain;
/// A fork chain - may not connect to genesis, must connect to a root chain at some point
/// Cant fully validate transactions without root chain. But can validate transactions within the fork itself
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
        *deltas.entry(*tx.receiver()).or_insert(INITIAL_DELTA) += tx.amount() as Delta;
        *deltas.entry(*tx.sender()).or_insert(INITIAL_DELTA) -= tx.amount() as Delta;
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
        let entry = deltas.entry(key).or_insert(INITIAL_DELTA);
        *entry += delta;
    }
}

/// Assumes deltas are valid (i.e. no negative balances will result)
fn apply_deltas_to_balances(balances: &mut AccountBalances, deltas: AccountDelta) {
    for (key, delta) in deltas {
        let entry = balances.entry(key).or_insert(INITIAL_BALANCE);
        // Safe to unwrap as we validated before
        let new_balance = ((*entry as Delta) + delta) as Balance;
        *entry = new_balance;
    }
}

fn recalculate_balances(blocks: &[Block]) -> HashMap<VerifyingKey, Balance> {
    assert_eq!(
        blocks
            .first()
            .expect("there is at least one block to recalc balances")
            .previous_hash(),
        &GENESIS_ROOT_HASH,
        "Can only recalc balances from genesis"
    );
    let mut balances = HashMap::new();
    for block in blocks {
        let deltas = get_transaction_deltas(block.transactions());
        assert!(
            validate_balance_deltas(&balances, &deltas),
            "Invalid balances in chain during recalculation"
        );
        apply_deltas_to_balances(&mut balances, deltas);
    }
    balances
}
fn verify_balances(balances: &HashMap<VerifyingKey, Balance>) -> bool {
    balances
        .values()
        .all(|&balance| balance + INITIAL_BALANCE >= 0)
}
// A valid chain must have at least one block
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound = "")]
pub(super) struct Chain<T: ChainType = RootChain> {
    /// Ordered sequence of blocks
    blocks: Vec<Block>,
    /// Must be valid non-negative balances.
    balances: HashMap<VerifyingKey, u64>,
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

    fn root_block(&self) -> &Block {
        self.blocks.first().expect("Chain has at least one block")
    }

    pub fn root_hash(&self) -> &Hash {
        self.root_block().hash()
    }

    fn get_balance(&self, address: &VerifyingKey) -> Option<Balance> {
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
        self.blocks.push(block);
        Ok(())
    }

    pub fn truncate_at(&mut self, hash: &Hash) -> Option<Chain<ForkChain>> {
        let index = self.find_block_index(hash)?;
        let split_blocks = self.blocks.split_off(index + 1);
        if split_blocks.is_empty() {
            // No blocks split, self.blocks remains unchanged
            return None;
        }
        // Recalculate balances up to the fork point
        let updated_root_balances = recalculate_balances(&self.blocks);

        // Balances for the fork chain will be the same as the root chain at the split point
        let fork_balances = std::mem::replace(&mut self.balances, updated_root_balances);

        Some(Chain {
            blocks: split_blocks,
            balances: fork_balances,
            _marker: PhantomData,
        })
    }
}

impl Chain<RootChain> {
    pub fn new_from_genesis(genesis: Block) -> Self {
        assert_eq!(genesis.previous_hash(), &GENESIS_ROOT_HASH);
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
        // Find the fork point
        let Some(index) = self.find_block_index(fork_chain.root_hash()) else {
            return Err(fork_chain);
        };
        let chain_cumulative_difficulty = self.cumulative_difficulty();
        let root_cumulative_difficulty = self.cumulative_difficulty_till(index);
        let fork_cumulative_difficulty = fork_chain.cumulative_difficulty();
        if root_cumulative_difficulty + fork_cumulative_difficulty <= chain_cumulative_difficulty {
            return Err(fork_chain);
        }
        // Truncate and append
        let Some(old_fork) = self.truncate_at(fork_chain.root_hash()) else {
            return Err(fork_chain);
        };
        // Swap in the fork chain
        self.append_fork(fork_chain);
        // Return the
        Ok(old_fork)
    }
}

impl Chain<ForkChain> {
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
        root_chain.contains_block(self.root_hash())
            && self.is_valid_against_root(root_chain.tip_hash(), &root_chain.balances)
    }

    /// Assumes the fork chain is valid against this root chain
    pub fn append_fork(&mut self, fork_chain: Self) {
        self.blocks.extend(fork_chain.blocks);
        self.balances = fork_chain.balances;
    }
}
