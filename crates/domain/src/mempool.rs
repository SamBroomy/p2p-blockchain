use std::collections::{HashSet, VecDeque};

use blake3::Hash;
use indexmap::IndexSet;

use crate::Transaction;

#[derive(Debug, Clone)]
pub struct Mempool {
    transactions: VecDeque<Transaction>,
    seen_hashes: HashSet<Hash>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
            seen_hashes: HashSet::new(),
        }
    }

    /// Add transaction if not already seen
    pub fn add_transaction(&mut self, tx: Transaction) -> bool {
        let tx_hash = tx.hash();
        if self.seen_hashes.insert(tx_hash) {
            self.transactions.push_back(tx);
            true
        } else {
            false
        }
    }

    /// Get the next transaction from the mempool
    pub fn get_transaction(&mut self) -> Option<Transaction> {
        self.transactions.pop_front()
    }

    pub fn remove_transactions(&mut self, txs: &[Transaction]) {
        let tx_hashes: IndexSet<Hash> = txs.iter().map(Transaction::hash).collect();
        self.transactions
            .retain(|tx| !tx_hashes.contains(&tx.hash()));
    }

    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }
}
