import hashlib

class ProofOfWork:
    def __init__(self, difficulty=3):
        """ 
        Constructor for the `ProofOfWork` class.
        :param block: Block to be mined.
        :param difficulty: Difficulty level for the proof of work.
        """
        self.difficulty = difficulty
        
    def mine(self, block):
        """ 
        Method to mine the block by finding a hash with the required number of leading zeroes.
        """
        block.nonce = 0
        guess_hash = block.compute_hash()
        while not guess_hash.startswith('0' * ProofOfWork.difficulty):
            block.nonce += 1
            guess_hash = block.compute_hash()
        return guess_hash
    
    @staticmethod
    def is_valid_proof(block, block_hash):
        """ 
        Method to check if the block_hash is a valid hash of the block and satisfies the difficulty criteria.
        :param block: Block to be checked.
        :param block_hash: Hash of the block to be checked.
        """
        return (block_hash.startswith('0' * ProofOfWork.difficulty) and block_hash == block.compute_hash())