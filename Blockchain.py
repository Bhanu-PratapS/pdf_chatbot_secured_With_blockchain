import random
from Block import Block

class Blockchain:
    difficulty = 3

    def __init__(self):
        self.pending_transactions = []
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        # Create the genesis block and append it to the chain
        genesis_block = Block(0, [], "0")
        genesis_block.hash = genesis_block.generate_hash()
        self.chain.append(genesis_block)

    def add_block(self, block, block_hash):
        previous_hash = self.last_block().hash
        if previous_hash == block.previous_hash and self.is_valid_proof(block, block_hash):
            block.hash = block_hash
            self.chain.append(block)
            return True
        return False

    def mine(self):
        if not self.pending_transactions:
            return False  # No transactions to mine

        last_block = self.last_block()
        new_block = Block(last_block.index + 1, self.pending_transactions, last_block.hash)
        
        # Perform proof of work to get the hash
        new_block.hash = self.proof_of_work(new_block)

        if self.add_block(new_block, new_block.hash):
            # If the block is successfully added, clear the transactions and return the new block
            self.pending_transactions = []
            return new_block  # Return the block itself instead of just the index
        return False

    def proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.generate_hash()
        while not computed_hash.startswith('0' * Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.generate_hash()
        return computed_hash

    def add_new_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def is_valid_proof(block, block_hash):
        return (block_hash.startswith('0' * Blockchain.difficulty) and
                block_hash == block.generate_hash())

    def check_chain_validity(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if not self.is_valid_proof(current, current.hash) or \
               current.previous_hash != previous.hash:
                return False
        return True
