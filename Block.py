from hashlib import sha256
from time import time

class Block:
    def __init__(self, index, transactions, previous_hash):
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.timestamp = time()  # Assign the current time to each block

    def generate_hash(self):
        # Combine block data into a single string and generate its hash
        block_contents = f"{self.index}{self.transactions}{self.previous_hash}{self.nonce}{self.timestamp}"
        return sha256(block_contents.encode()).hexdigest()

    def add_transaction(self, transaction):
        if isinstance(self.transactions, list):
            self.transactions.append(transaction)
        else:
            raise ValueError("Transactions must be a list")

    def __repr__(self):
        return f"Block(index={self.index}, transactions={self.transactions}, previous_hash='{self.previous_hash}', nonce={self.nonce}, timestamp={self.timestamp})"

# Example usage
if __name__ == "__main__":
    # Create a block
    block = Block(index=0, transactions=[], previous_hash="0")

    # Add a transaction
    block.add_transaction("userA sends 2 BTC to userB")

    # View the block's data
    print(block)
