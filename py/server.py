import hashlib
import json
import time
import sys
from flask import Flask, jsonify, request
from uuid import uuid4
import requests 
import random
import pickle
from blockchain.blockchain import Block, Blockchain
from blockchain.transaction import Transaction, TransactionPool
from threading import Thread, Event
from federated_learning.client import Worker
from federated_learning.model.simpleCNN import SimpleCNN    


class PoWThread(Thread):
    def __init__(self, stop_event, blockchain, node):
        self.stop_event = stop_event    
        Thread.__init__(self)
        self.blockchain = blockchain    
        self.node = node
        self.response = None    
    
    def run(self):
        block, stopped = self.blockchain.proof_of_work(self.stop_event)
        self.response = {
            'message': "End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)
        
        
STOP_EVENT = Event()

app = Flask(__name__)
status = {
    's': "receiving",
    'id': str(uuid4()).replace('-', ''),
    'blockchain': None,
    'address': ""
}

def mine():
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT, status['blockchain'], status['id'])
    status['s'] = "mining"
    thread.start()

def on_end_mining(stopped):
    if status['s'] == 'receiving':
        return
    if stopped:
        status['blockchain'].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status['blockchain'].nodes:
        requests.get('http://{node}/stop_mining'.format(node=node))
    
@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    """ 
    send a new transaction to the transaction pool.
    """
    if status['s'] != "receiving":
        return 'Miner not ready', 400
    values = request.get_json()
    required = ['sender_id', 'task', 'model_params', 'accuracy', 'timestamp']
    if not all(k in values for k in required):
        return 'Missing values', 400
    transaction = Transaction(
        values['sender_id'], 
        values['task'], 
        values['model_params'], 
        values['accuracy'], 
        values['timestamp']
    )
    status['blockchain'].add_transaction(transaction)
    return 'Transaction added to the pool', 200

@app.route('/transactions/pool', methods=['GET'])
def get_transactions():
    transactions = status['blockchain'].transaction_pool.get_transactions()
    return jsonify(transactions), 200   

@app.route('/status', methods=['GET'])
def get_status():
    response = {
        'status': status['s'],
        'last_block_index': status['blockchain'].last_block.index if status['blockchain'] else 0,
    }
    return jsonify(response), 200

@app.route('/chain', methods=['GET'])   
def get_chain():
    chain_data = []
    for block in status['blockchain'].chain:
        chain_data.append(block.__dict__)
        chain_data[-1]['transactions'] = block.transactions.__repr__()
    return jsonify(chain_data), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400
    for node in nodes:
        if node != status['address'] and node not in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
        for miner in status['blockchain'].nodes:
        # assume for now that all nodes will mine
            requests.post('http://{miner}/nodes/register'.format(miner=miner), json={'nodes': [node]})
    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(status['blockchain'].nodes),
    }
    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = status['blockchain'].resolve_conflicts(STOP_EVENT)
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': status['blockchain'].chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': status['blockchain'].chain
        }
    return jsonify(response), 200

@app.route('/stop_mining', methods=['GET']) 
def stop_mining():
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'message': 'Mining stopped'
    }
    return jsonify(response), 200
    
if __name__ == '__main__':
    base_model = SimpleCNN()
    port = 5000
    status['blockchain'] = Blockchain(base_model=base_model)
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)