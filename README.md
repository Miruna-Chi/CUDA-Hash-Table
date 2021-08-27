# CUDA-Hash-Table
A thread-safe Hash Table using Nvidia’s API for CUDA-enabled GPUs. 200 times faster than the C++ only code through sheer exploitation of a GPU’s fine-grained parallelism.

# HashTable Idea
- pair vector: key - value

# Adding an item
- GPU function, for each key-value pair in the lists given by insertBatch:
  * If the key and value are valid, proceed to the next step
  
  * Make the hash using the Multiplication Method: hash = capacity * (fmod (key * A, 1.0));
    where A is a constant equal to 0.5 × (√5 - 1)
    
  * Resolve collisions (if any):
    - atomicCAS (address, val_comp, new_val)
    Each thread evaluates the hash of an element to an index, then linearly searches for a key match to the right, until the end of the array and then
    starts over, if needed. It stops at the first free slot where it could put the element (or earlier if it finds the key already set to a value from a previous batch). 
    If each thread will do this, we need the operation of checking/modifying/adding an element to be thread-safe (atomic and serializable). This is where atomicCAS comes in.
    
  * In case a new element is placed on a free position (not an update), the variable **slots_occ**
  existing on both Host and GPU is incremented.
 
# Extracting an item
- searches the buckets in the same manner as the previous operation, no need for serialization (nothing is written)

# Copying keys and resizing values
- copy a batch of keys and values from the **hash_table** (used for resizing)

# Reshape
- allocate a batch of keys of **size = hash_table capacity**
- copies them via kernel to the device
- frees memory for the **hash_table**
- creates a new **hash_table** with the **new capacity**
- calls insertBatch with the batch from the previous steps to fill the new **hash_table** with the previous values

# insertBatch
- see code comments for a detailed explanation of how the new capacity is calculated
- calls reshape if the **load_factor** exceeds max limit
- allocates memory, adds elements, frees memory

# getBatch
- allocates memory, calls getBatch, frees memory

