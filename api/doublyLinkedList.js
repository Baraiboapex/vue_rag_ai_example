class Node {
    next = null;
    prev = null;
    data = null;
  
    constructor(data) {
      this.data = data;
    }
  }
  
  class DoublyLinkedList {
    head = null;
    tail = null;
  
    push(itemData) {
      const currentNode = new Node(itemData);
  
        if (!this.head) {
          this.head = currentNode;
          this.tail = currentNode;
    
          return;
        }
        
        this.tail.next = currentNode;
        currentNode.prev = this.tail;
        this.tail = currentNode;
    }
    shift() {
      if (this.head === null)
        return this.listError("List does not have a head node");
  
      this.head = this.head.next;
  
      if (this.head) {
        this.head.prev = null;
      } else {
        this.head = null;
      }
    }
    pop() {
      if (this.tail === null)
        return this.listError("List does not have a tail node");
  
      this.tail = this.tail.prev;
  
      if (this.head) {
        this.head.null = null;
      } else {
        this.head = null;
      }
    }
    readList() {
      let currentNode = this.head;
  
      while (currentNode) {
        console.log(currentNode);
        currentNode = currentNode.next;
      }
    }
    getHeadNode() {
      return this.head;
    }
    getTailNode() {
      return this.tail;
    }
    listError(err) {
      console.error(err);
    }
  }
  
  module.exports = DoublyLinkedList;
  