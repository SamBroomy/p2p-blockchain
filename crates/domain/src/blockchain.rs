use blake3::Hash;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};

use crate::block::{Block, BlockConstructor};

fn is_previous_block_valid(previous_block: &Block, new_block: &Block) -> bool {
    previous_block.hash() == new_block.previous_hash()
}

// Todo: handle when we get competing block / chains from peers.
// An invariant should be that the chain with the most cumulative difficulty is the valid one
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockChain {
    pub chain: Vec<Block>,
    pub difficulty: usize,
}

impl BlockChain {
    fn create_genesis_block(difficulty: usize) -> Block {
        BlockConstructor::new(0, &[], Hash::from_bytes([0u8; 32])).mine(difficulty, None)
    }

    pub fn new(difficulty: usize) -> Self {
        let genesis_block = Self::create_genesis_block(difficulty);
        Self {
            chain: vec![genesis_block],
            difficulty,
        }
    }

    pub fn latest_block(&self) -> &Block {
        self.chain.last().expect("always be at least one block")
    }

    pub fn add_block(&mut self, block: Block) -> Result<(), &'static str> {
        if !block.is_valid() {
            return Err("Invalid block");
        }
        if block.previous_hash() != self.latest_block().hash() {
            return Err("Previous hash does not match");
        }
        self.chain.push(block);
        Ok(())
    }

    pub fn is_chain_valid(&self) -> bool {
        self.chain.windows(2).all(|w| {
            // don't need to check for single (genesis) block case, windows(2) returns empty iterator which makes all() return true
            let previous_block = &w[0];
            let current_block = &w[1];

            is_previous_block_valid(previous_block, current_block)
            // maybe don't need these checks as blocks themselves are validated on creation
                && current_block.is_valid()
                && previous_block.is_valid()
        })
    }

    /// Calculate balance for an address by scanning the entire chain
    pub fn get_balance(&self, address: &VerifyingKey) -> u64 {
        // For simplicity everyone starts with a balance of 100
        let mut balance = 100u64;

        for block in &self.chain {
            for tx in block.transactions() {
                if tx.receiver() == address {
                    balance = balance.saturating_add(tx.amount());
                }
                if tx.sender() == address {
                    balance = balance.saturating_sub(tx.amount());
                }
            }
        }

        balance
    }

    pub fn cumulative_difficulty(&self) -> u128 {
        self.chain
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

    // TODO: implement fork resolution. I think what I want to do is have some type `Chain` or something like that that can represent a subsection of blocks that way we can have multiple competing chains and then pick the one with the highest cumulative difficulty.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blockchain_creation() {
        let blockchain = BlockChain::new(1);

        assert_eq!(blockchain.chain.len(), 1);
        assert!(blockchain.is_chain_valid());
    }

    #[test]
    fn test_add_valid_block() {
        let mut blockchain = BlockChain::new(1);
        let latest_hash = *blockchain.latest_block().hash();

        let constructor = BlockConstructor::new(1, &[], latest_hash);
        let new_block = constructor.mine(1, None);

        assert!(blockchain.add_block(new_block).is_ok());
        assert_eq!(blockchain.chain.len(), 2);
        assert!(blockchain.is_chain_valid());
    }

    #[test]
    fn test_reject_invalid_previous_hash() {
        let mut blockchain = BlockChain::new(1);

        let wrong_hash = Hash::from_bytes([99u8; 32]);
        let constructor = BlockConstructor::new(1, &[], wrong_hash);
        let new_block = constructor.mine(1, None);

        assert!(blockchain.add_block(new_block).is_err());
        assert_eq!(blockchain.chain.len(), 1);
    }

    #[test]
    fn test_blockchain_with_transactions() {
        use crate::wallet::Wallet;

        let mut blockchain = BlockChain::new(1);
        let mut wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 100);
        let latest_hash = *blockchain.latest_block().hash();

        let constructor = BlockConstructor::new(1, &[tx], latest_hash);
        let new_block = constructor.mine(1, None);

        assert!(blockchain.add_block(new_block).is_ok());
        assert!(blockchain.is_chain_valid());
    }

    #[test]
    fn test_blockchain_serialization() {
        let blockchain = BlockChain::new(1);

        let serialized = serde_json::to_string(&blockchain).expect("serialization failed");
        let deserialized: BlockChain =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(blockchain.chain.len(), deserialized.chain.len());
        assert_eq!(blockchain.difficulty, deserialized.difficulty);
        assert!(deserialized.is_chain_valid());
    }
}
