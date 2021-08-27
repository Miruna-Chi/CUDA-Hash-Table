#include <iostream>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

// constant	for the Hash Function (using the Multiplication Method)
__constant__ double	A = 0.618033989;  // which is 0.5×(√5 − 1)

// shared variable - the devices and host can see it
__managed__ unsigned int slots_occ = 0;

__global__ void addElement(HashTable hash_table, int capacity,
	float load_factor, int *keys, int *values, int numKeys) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Avoid accessing out of bounds elements && key and value are valid
	if (index < numKeys) {
		unsigned int key = keys[index];
		unsigned int value = values[index];
		
		if (key > 0 && value > 0) {
			unsigned int hash = capacity * (fmod(key * A, 1.0));

			// proceed with linearProbing
			unsigned int counter = 0;

			// try to put the key in the table
			
			while (counter < capacity) {

			if (atomicCAS(&(hash_table[hash].key), key, key) == key) {
				break;
			}
			if (atomicCAS(&(hash_table[hash].key), 0, key) == 0) {
				atomicAdd(&slots_occ, 1);
				break;
			}
				counter++;
				hash++;
				if(hash >= capacity)
					hash = 0;
			}

			// something went wrong, no space for new kv_pair
			if (counter == capacity) {
			//	printf("really?\n");
				return;
			}

			// associate value
			hash_table[hash].value = value;
		}
	}
}

__global__ void getElement(HashTable hash_table, int capacity,
	int *keys, int *values, int numKeys) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	

	// Avoid accessing out of bounds elements && key and value are valid
	if (index < numKeys) {
		unsigned int key = keys[index];
		if (key > 0) {
			unsigned int hash = capacity * fmod(key * A, 1.0);

			// find elem
			if (hash_table[hash].key != key) {
				// key not found yet

				// If it's the last position in the array, go to the beginning
				if (hash == capacity - 1)
					hash = 0;

				// proceed with linearChecking

				unsigned int counter = 0;

				// try finding the key

				while (hash_table[hash].key != key && counter < capacity) {
					counter++;
					hash++;
					if(hash >= capacity)
						hash = 0;
				}

				// key not found
				if (counter == capacity) {
					return;
				}
				else values[index] = hash_table[hash].value;
			}

			values[index] = hash_table[hash].value;
		}
		else {
			values[index] = -1;
		}
	}
}

__global__ void copyKeyVal(int *keys, int *values, int capacity, HashTable hash_table) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < capacity) {
		keys[index] = hash_table[index].key;
		values[index] = hash_table[index].value;
	} 
}

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/



__global__ void init_kv_pairs(HashTable hash_table, int capacity) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < capacity) {
		hash_table[index].key = 0;
		hash_table[index].value = 0;
	} 
}

void GpuHashTable::initAttr(int size) {
	hash_table = 0;

	glbGpuAllocator->_cudaMallocManaged((void **) &hash_table, size * sizeof(struct kv_pair));

	if (hash_table == 0) {
		perror("initAttr: cudaMallocManaged failed");
		exit(1);
	}

	size_t block_size = 256;
	size_t blocks_no = size / block_size;
 
	// take care of a possible partially occupied block
	if (size % block_size) 
		++blocks_no;

	init_kv_pairs<<<blocks_no, block_size>>>(hash_table, size);
	cudaDeviceSynchronize();

	capacity = size;
	slots_occ = 0;
	slots_occupied = 0;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {

	if (size <= 0) {
		perror("GpuHashTable: size has to be > 0");
		exit(1);
	}

	initAttr(size);
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(hash_table);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	int *keys;
	int *values;

	int *host_keys = (int *)malloc(sizeof(int) * capacity);
	int *host_values = (int *)malloc(sizeof(int) * capacity);

	glbGpuAllocator->_cudaMalloc((void **) &keys, sizeof(int) * capacity);
	glbGpuAllocator->_cudaMalloc((void **) &values, sizeof(int) * capacity);

	if (keys == 0 || values == 0) {
		perror("initAttr: cudaMallocManaged failed");
		exit(1);
	}

	size_t block_size = 256;
	size_t blocks_no = capacity / block_size;
 
	// take care of a possible partially occupied block
	if (capacity % block_size) 
		++blocks_no;

	
	copyKeyVal<<<blocks_no, block_size>>>(keys, values, capacity, hash_table);
	cudaDeviceSynchronize();

	cudaMemcpy(host_keys, keys, sizeof(int) * capacity, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_values, values, sizeof(int) * capacity, cudaMemcpyDeviceToHost);


	glbGpuAllocator->_cudaFree(hash_table);
	glbGpuAllocator->_cudaFree(keys);
	glbGpuAllocator->_cudaFree(values);
	
	int old_capacity = capacity;
	initAttr(numBucketsReshape);

	if (!insertBatch(host_keys, host_values, old_capacity)) {
		perror("reshape: failed");
		exit(1);
	}

	free(host_keys);
	free(host_values);

}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	/*
	 * 		When load_factor * capacity is reached, capacity should increase
	 * by resize_factor at a new element's addition. So the new fill would be:
	 *			(load_factor / resize_factor)
	 * Now, that's for one new element, but we're adding numKeys elements. 
	 * 		If slots_occupied + no_of_new_elements exceeds that load, then
	 * we shouldn't just double the capacity, because we might need to do so
	 * repeatedly and waste resources. So we're going to consider a temporary
	 * capacity for which (slots_occupied + numKeys) represents the
	 * load_factor %. That is what we're going to double: temp_capacity.
	 * 		Eg: load_factor = 0.75
	 * 		slots_occupied + numKeys >= 0.75 * temp_capacity =>
	 *		=> temp_capacity = (slots_occupied + numKeys) / 0.75
	 * 		new_capacity = resize_factor * temp_capacity
	 */
	if (slots_occupied + numKeys >= load_factor * capacity) {
		int new_capacity = resize_factor * (slots_occupied + numKeys) / load_factor;
		reshape(new_capacity);
	}
	size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;
 
	// take care of a possible partially occupied block
	if (numKeys % block_size) 
		++blocks_no;

	int *device_keys;
	int *device_values;

	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	if(device_values == 0 || device_keys == 0) {
		perror("insertBatch: cudaMalloc failed");
		exit(1);
	}

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);


	addElement<<<blocks_no, block_size>>>(hash_table, capacity,
		load_factor, device_keys, device_values, numKeys);
	
	// wait for kernel to finish
	cudaDeviceSynchronize();

	slots_occupied = slots_occ;

	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *device_values;

	int *values = (int *)malloc(numKeys * sizeof(int));

	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	

	if(device_values == 0 || device_keys == 0 || values == 0) {
		perror("insertBatch: cudaMalloc failed");
		exit(1);
	}

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;
 
	// take care of a possible partially occupied block
	if (numKeys % block_size) 
		++blocks_no;

	getElement<<<blocks_no, block_size>>>(hash_table, capacity, device_keys,
		device_values, numKeys);

	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);

	return values;
}