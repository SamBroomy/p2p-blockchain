use std::error::Error;

use blockchain_types::wallet::{Address, Wallet};
use futures::StreamExt;
use node::Node;
use tokio::io::{self, AsyncBufReadExt};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // Default: info level, but suppress mDNS interface errors
                "info,libp2p_mdns::behaviour::iface=off".into()
            }),
        )
        .init();
    let args: Vec<String> = std::env::args().collect();

    // Get wallet seed from args or generate random
    let seed = args.get(1).map(String::as_str);
    let wallet = seed.map_or_else(
        || {
            warn!("⚠️  No seed provided, generating random wallet");
            Wallet::new()
        },
        Wallet::from_seed,
    );

    let mut node = Node::new(wallet)?;

    info!("🔐 Your address: {}", node.my_address());
    info!("💰 Balance: {:?}", node.get_balance());
    info!("");
    info!("Commands:");
    info!("  balance              - Show your balance");
    info!("  send <addr> <amount> - Send coins to an address");
    info!("  mine                 - Mine a block");
    info!("  status               - Show blockchain status");

    // Read commands from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();

    loop {
        tokio::select! {
            // Handle network events
            event = node.swarm.select_next_some() => {
                node.handle_swarm_event(event);
            }

            // Handle user commands
            Ok(Some(line)) = stdin.next_line() => {
                let parts: Vec<&str> = line.split_whitespace().collect();

                match parts.as_slice() {
                    ["balance"] => {
                        info!("💰 Balance: {:?}", node.get_balance().unwrap_or(100));
                    }

                    ["send", addr_str, amount_str] => {
                        let amount: u64 = amount_str.parse()?;
                        let address = Address::from_hex(*addr_str)?;

                        match node.send_transaction(&address, amount) {
                            Ok(tx) => info!("✅ Sent {} coins (tx: {:?})", amount, tx.hash()),
                            Err(e) => info!("❌ Error: {e}"),
                        }
                    }

                    // ["send-to", name, amount_str] => {
                    //     let amount: u64 = amount_str.parse()?;
                    //     match node.send_to(name, amount) {
                    //         Ok(tx) => println!("✅ Sent {} coins to {name}", amount),
                    //         Err(e) => println!("❌ Error: {e}"),
                    //     }
                    // }

                    // ["add", name, addr_str] => {
                    //     let address = Address::from_hex(addr_str)?;
                    //     node.add_contact(name.to_string(), address);
                    // }

                    // ["contacts"] => {
                    //     for (name, addr) in node.list_contacts() {
                    //         println!("  {name}: {addr}");
                    //     }
                    // }

                    ["mine"] => {
                        info!("⛏️  Mining...");
                        match node.mine_block() {
                            Ok(block) => info!("✅ Mined block {:?}", block.hash()),
                            Err(e) => info!("❌ Error: {e}"),
                        }
                    }

                    ["status"] => {
                        info!("📊 Blockchain Status:");
                        info!("   Height: {}", node.blockchain.height());
                        info!("   Difficulty: {}", node.blockchain.difficulty);
                        info!("   Peers: {:#?}", node.list_peers().collect::<Vec<_>>());
                        info!("   Mempool: {} txs", node.mempool.len());
                        info!("   Orphans: {}", node.blockchain.orphan_count());
                    }

                    _ => {
                        info!("❌ Unknown command");
                    }
                }
            }
        }
    }
}
