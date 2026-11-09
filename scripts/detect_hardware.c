#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <omp.h>
#include <cpuid.h>

void get_cpu_brand(char *brand) {
    unsigned int regs[4];
    for (unsigned int i = 0; i < 3; ++i) {
        __cpuid(0x80000002 + i, regs[0], regs[1], regs[2], regs[3]);
        memcpy(brand + i * 16, regs, 16);
    }
    brand[48] = '\0';
}

int main() {
    char brand[49] = {0};
    get_cpu_brand(brand);
    printf("CPU Brand: %s\n", brand);
    printf("Max OpenMP Threads: %d\n", omp_get_max_threads());

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, NULL, &len);
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
    if (GetLogicalProcessorInformationEx(RelationAll, buffer, &len)) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX curr = buffer;
        DWORD offset = 0;
        int core_count = 0;
        int numa_count = 0;
        size_t l1d_size = 0, l1i_size = 0, l2_size = 0, l3_size = 0;

        while (offset < len) {
            if (curr->Relationship == RelationProcessorCore) {
                core_count++;
            } else if (curr->Relationship == RelationNumaNode) {
                numa_count++;
            } else if (curr->Relationship == RelationCache) {
                CACHE_RELATIONSHIP *cache = &curr->Cache;
                if (cache->Level == 1) {
                    if (cache->Type == CacheData) l1d_size = cache->CacheSize;
                    else if (cache->Type == CacheInstruction) l1i_size = cache->CacheSize;
                } else if (cache->Level == 2) {
                    l2_size = cache->CacheSize;
                } else if (cache->Level == 3) {
                    l3_size = cache->CacheSize;
                }
            }
            offset += curr->Size;
            curr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((char*)buffer + offset);
        }
        printf("Physical Cores: %d\n", core_count);
        printf("NUMA Nodes: %d\n", numa_count);
        printf("L1 Data Cache (per core): %zu KB\n", l1d_size / 1024);
        printf("L2 Cache: %zu KB\n", l2_size / 1024);
        printf("L3 Cache (LLC): %zu MB\n", l3_size / (1024 * 1024));
    }
    free(buffer);

    // Check SIMD extensions via cpuid
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    int has_fma = (ecx & (1 << 12)) != 0;
    int has_avx = (ecx & (1 << 28)) != 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    int has_avx2 = (ebx & (1 << 5)) != 0;
    int has_avx512f = (ebx & (1 << 16)) != 0;

    printf("Features: AVX=%d, AVX2=%d, FMA=%d, AVX512F=%d\n", has_avx, has_avx2, has_fma, has_avx512f);
    return 0;
}

