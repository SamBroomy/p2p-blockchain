use domain::{BlockChain, BlockConstructor, Wallet};

fn main() {
    println!("🔗 P2P Blockchain Starting...\n");

    // Initialize blockchain
    let mut blockchain = BlockChain::new(6);
    println!(
        "✅ Blockchain initialized with difficulty {}",
        blockchain.difficulty
    );
    println!("   Genesis block: {:?}\n", blockchain.latest_block().hash());
    dbg!(&blockchain);

    // Create wallets
    let mut alice = Wallet::from_seed("alice");
    let bob = Wallet::from_seed("bob");

    println!("👛 Wallets created:");
    println!("   Alice: {:?}", alice.address());
    dbg!(&alice);
    println!("   Bob:   {:?}\n", bob.address());
    dbg!(&bob);

    // Create and mine transaction
    println!("💸 Creating transaction: Alice -> Bob (100 coins)");
    let tx = alice.create_transaction(bob.address(), 100);
    dbg!(&tx);

    println!("⛏️  Mining block...");
    // Add block with transaction
    let block = BlockConstructor::new(1, &[tx], *blockchain.latest_block().hash())
        .mine(blockchain.difficulty, None);
    blockchain.add_block(block).expect("Failed to add block");

    println!("\n✅ Blockchain running!");
    println!("   Total blocks: {}", blockchain.chain.len());
    dbg!(&blockchain);
    println!("Is blockchain valid? {}", blockchain.is_chain_valid());
}
