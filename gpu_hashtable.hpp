#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

typedef struct kv_pair {
	int key;
	int value;
} *HashTable;

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		HashTable hash_table;
		unsigned int capacity;
		unsigned int slots_occupied;
		float load_factor = 0.9f;
		float resize_factor = 1.5f;


		GpuHashTable(int size);
		void initAttr(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
	
		~GpuHashTable();
};

#endif

