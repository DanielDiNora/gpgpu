#import <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

void cuda_check(cudaError_t err, char const *str)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s: CUDA error %d (%s)\n",
			str, err, cudaGetErrorString(err));
	}
}

__global__
void init_vec(int nels, float* __restrict__ d_vec1)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	d_vec1[i] = i;
}

__global__
void multi_vec(int n_row1,int n_col1,int n_row2,int n_col2,float* __restrict__ res_vec,float* __restrict__ d_vec1,float* __restrict__ d_vec2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int c= blockIdx.x*n_row1 + (threadIdx.x)%n_col1;
	int j= ((int)(threadIdx.x/n_row2) + (threadIdx.x%n_row2)*n_col2);
	res_vec[i]=d_vec1[c]*d_vec2[j];

}

__global__
void scalareMatrice(float* __restrict__ res_vec,float scalar,float* __restrict__ d_vec)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	res_vec[i]=d_vec[i]*scalar;
}

__global__
void reduction_row(int N,float* __restrict__ res_vec,float* __restrict__ d_vec1)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int idx=(int)(i/N);
	float c =res_vec[idx];
	float d =d_vec1[i];
	if(i%N==31){
		res_vec[idx]=d_vec1[i-31]+d_vec1[i-30]+d_vec1[i-29]+d_vec1[i-28]+d_vec1[i-27]+d_vec1[i-26]+d_vec1[i-25]+d_vec1[i-24]+
								d_vec1[i-23]+d_vec1[i-22]+d_vec1[i-21]+d_vec1[i-20]+d_vec1[i-19]+d_vec1[i-18]+d_vec1[i-17]+d_vec1[i-16]+
								d_vec1[i-15]+d_vec1[i-14]+d_vec1[i-13]+d_vec1[i-12]+d_vec1[i-11]+d_vec1[i-10]+d_vec1[i-9]+d_vec1[i-8]+
								d_vec1[i-7]+d_vec1[i-6]+d_vec1[i-5]+d_vec1[i-4]+d_vec1[i-3]+d_vec1[i-2]+d_vec1[i-1]+d_vec1[i];
	}
}

__global__
void transpose(int nrow,int ncols, float* __restrict__ res_vec, float* __restrict__ d_vec1)
{
	int c = threadIdx.x;
	int r=blockIdx.x;
	int l_in = r*ncols + c;
	int l_out = c * nrow + r;
	res_vec[l_out] = d_vec1[l_in];
}


__global__
void vecsum(int nels, float* __restrict__ res_vec, float* __restrict__ d_vec1, float* __restrict__ d_vec2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	res_vec[i] = d_vec1[i]+d_vec2[i];
}

__global__
void vecdif(int nels, float* __restrict__ res_vec, float* __restrict__ d_vec1, float* __restrict__ d_vec2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	res_vec[i] = d_vec1[i]-d_vec2[i];
}
void stampa(float* matrice,int m){
  int i,j;
	printf("\n");
  for(i=0;i<m;i++){
      printf("%f  ",matrice[i]);
			printf("\n");
	}

}

int main(int argc, char *argv[]){
  float* matriceA;
  float* matriceB;
  float* matriceX;
	float* pk;
	float* trasposta;
	float* prodotto;
	float* somma;
	float* res;
	float* den;
	float* res0;
	float* res1;
	float* res2;
	float* red_den;
  float* matrice;
	float* scalar;
	float* num;
	float* deno;
  float ak;
  int nels;
  printf("%d\n",argc );
  if (argc != 2) {
    error("syntax: serve N come arg");
  }

  int N = atoi(argv[1]);
  if (N < 0) {
    error("N < 0");
  }
	int M=1;

  nels=N*N;
  size_t memsize = nels*sizeof(float);
  cudaError_t err;
  err = cudaMalloc((void**)&matriceA, memsize);
  cuda_check(err, "alloc matriceA");
  err = cudaMalloc((void**)&matriceB, N*M*sizeof(float));
  cuda_check(err, "alloc matriceB");
  err = cudaMalloc((void**)&matriceX, N*sizeof(float));
  cuda_check(err, "alloc matriceX");
  err = cudaMallocHost(&matrice, N*N*sizeof(float));
	cuda_check(err, "alloc matrice");
	err = cudaMallocHost(&num, M*sizeof(float));
	cuda_check(err, "alloc matrice");
	err = cudaMallocHost(&deno, M*sizeof(float));
	cuda_check(err, "alloc matrice");
	err = cudaMalloc((void**)&somma,nels*M*sizeof(float));
	cuda_check(err, "alloc somma");
	err = cudaMalloc((void**)&res,M*N*N*sizeof(float));
	cuda_check(err, "alloc res");
	err = cudaMalloc((void**)&res0,M*N*sizeof(float));
	cuda_check(err, "alloc res0");
	err = cudaMalloc((void**)&prodotto,M*N*N*sizeof(float));
	cuda_check(err, "alloc prodotto");
	err = cudaMalloc((void**)&res1,M*N*sizeof(float));
	cuda_check(err, "alloc res1");
	err = cudaMalloc((void**)&res2,M*N*sizeof(float));
	cuda_check(err, "alloc res2");
	err = cudaMalloc((void**)&pk,M*N*sizeof(float));
	cuda_check(err, "alloc pk");
	err = cudaMalloc((void**)&trasposta,M*N*sizeof(float));
	cuda_check(err, "alloc trasposta	");
	err = cudaMalloc((void**)&den,M*N*sizeof(float));
	cuda_check(err, "alloc den");
	err = cudaMalloc((void**)&red_den,M*sizeof(float));
	cuda_check(err, "alloc den");
	err = cudaMalloc((void**)&scalar,M*N*sizeof(float));
	cuda_check(err, "alloc scalar");

  cudaEvent_t pre_init, post_init, pre_sum, post_sum, pre_red, post_red,pre_prodotto,post_prodotto,
							pre_transpose,post_transpose,pre_scalar_matrice,post_scalar_matrice,pre_vecsum,post_vecsum,
							pre_vecdif,post_vecdif;


	err = cudaEventCreate(&pre_init, 0);
	cuda_check(err, "create pre_init");
	err = cudaEventCreate(&pre_red, 0);
	cuda_check(err, "create pre_red");
	err = cudaEventCreate(&pre_prodotto, 0);
	cuda_check(err, "create pre_sum");
	err = cudaEventCreate(&pre_transpose, 0);
	cuda_check(err, "create pre_traspose");
	err = cudaEventCreate(&pre_scalar_matrice, 0);
	cuda_check(err, "create pre_scalar_matrice");
	err = cudaEventCreate(&pre_vecdif, 0);
	cuda_check(err, "create pre_vecdif");
	err = cudaEventCreate(&pre_vecsum, 0);
	cuda_check(err, "create pre_vecsum");

	err = cudaEventCreate(&post_init, 0);
	cuda_check(err, "create post_init");
	err = cudaEventCreate(&post_red, 0);
	cuda_check(err, "create post_red");
	err = cudaEventCreate(&post_prodotto, 0);
	cuda_check(err, "create post_sum");
	err = cudaEventCreate(&post_transpose, 0);
	cuda_check(err, "create post_traspose");
	err = cudaEventCreate(&post_scalar_matrice, 0);
	cuda_check(err, "create post_scalar_matrice");
	err = cudaEventCreate(&post_vecdif, 0);
	cuda_check(err, "create post_vecdif");
	err = cudaEventCreate(&post_vecsum, 0);
	cuda_check(err, "create post_vecsum");


	cudaEventRecord(pre_init);
	init_vec<<<N, N>>>(nels, matriceA);
	cudaEventRecord(post_init);
  init_vec<<<1, M*N>>>(M*N, matriceB);
	init_vec<<<1, M*N>>>(M*N, matriceX);
	int i;

	for(i=0;i<1;i++){
		cudaEventRecord(pre_prodotto);
		multi_vec<<<N, M*N>>>(N,N,N,M,somma,matriceA,matriceX);
		cudaEventRecord(post_prodotto);
		cudaEventRecord(pre_red);
		reduction_row<<<N, M*N>>>(N,res0,somma);
		cudaEventRecord(post_red);

		cudaEventRecord(pre_vecdif);
		vecdif<<<N,M>>>(N*M,pk,matriceB,res0);
		cudaEventRecord(post_vecdif);

		cudaEventRecord(pre_transpose);
		transpose<<<N,M>>>(N,M,trasposta,pk);
		cudaEventRecord(post_transpose);

		multi_vec<<<M, N>>>(M,N,N,M,prodotto,trasposta,pk);
		reduction_row<<<M, N>>>(N,res1,prodotto);

		multi_vec<<<M, M*N*N>>>(M,N,N,N,res,trasposta,matriceA);
		reduction_row<<<M*N, N>>>(N,res2,res);
		multi_vec<<<N, M*N>>>(M,N,N,M,den,res2,pk);
		reduction_row<<<N, M*N>>>(N,red_den,den);

		err = cudaMemcpy(num, res1, 1*sizeof(float), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(deno, red_den, 1*sizeof(float), cudaMemcpyDeviceToHost);
		ak=num[0]/deno[0];

		cudaEventRecord(pre_scalar_matrice);
		scalareMatrice<<<N, M>>>(scalar,ak,pk);
		cudaEventRecord(post_scalar_matrice);

		cudaEventRecord(pre_vecsum);
		vecsum<<<N, M>>>(N*M*N,matriceX,matriceX,scalar);
		cudaEventRecord(post_vecsum);

	  err = cudaMemcpy(matrice, matriceX, M*N*sizeof(float), cudaMemcpyDeviceToHost);
		cuda_check(err, "create mem");
	  stampa(matrice,M*N);



		float runtime_init_ms, runtime_prodotto_ms, runtime_red_ms,runtime_transpose_ms,runtime_scalar_matrice_ms,
					runtime_vecdif_ms,runtime_vecsum_ms;
		err = cudaEventElapsedTime(&runtime_init_ms, pre_init, post_init);
		cuda_check(err, "elapsed time init");
		err = cudaEventElapsedTime(&runtime_prodotto_ms, pre_prodotto, post_prodotto);
		cuda_check(err, "elapsed time prodotto");
		err = cudaEventElapsedTime(&runtime_red_ms, pre_red, post_red);
		cuda_check(err, "elapsed time reduction");
		err = cudaEventElapsedTime(&runtime_transpose_ms, pre_transpose, post_transpose);
		cuda_check(err, "elapsed time traspose");
		err = cudaEventElapsedTime(&runtime_scalar_matrice_ms, pre_scalar_matrice, post_scalar_matrice);
		cuda_check(err, "elapsed time scalar_matrice");
		err = cudaEventElapsedTime(&runtime_vecdif_ms, pre_vecdif, post_vecdif);
		cuda_check(err, "elapsed time vecdif");
		err = cudaEventElapsedTime(&runtime_vecsum_ms, pre_vecsum, post_vecsum);
		cuda_check(err, "elapsed time vecsum");


		printf("init: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_init_ms, nels/runtime_init_ms/1.0e6, memsize/runtime_init_ms/1.0e6);
		printf("prodotto: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_prodotto_ms, nels/runtime_prodotto_ms/1.0e6, memsize/runtime_prodotto_ms/1.0e6);
		printf("reduction: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_red_ms, nels/runtime_red_ms/1.0e6, memsize/runtime_red_ms/1.0e6);
		printf("transpose: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_transpose_ms, N/runtime_transpose_ms/1.0e6, (N*sizeof(float))/runtime_transpose_ms/1.0e6);
		printf("scalareMatrice: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_scalar_matrice_ms, N/runtime_scalar_matrice_ms/1.0e6, (N*sizeof(float))/runtime_scalar_matrice_ms/1.0e6);
		printf("vecdif: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
			runtime_vecdif_ms, N/runtime_vecdif_ms/1.0e6, (N*sizeof(float))/runtime_vecdif_ms/1.0e6);
		printf("vecsum: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
				runtime_vecsum_ms, N/runtime_vecsum_ms/1.0e6, (N*sizeof(float))/runtime_vecsum_ms/1.0e6);
	}
  cudaFree(matriceA);
	cudaFreeHost(matrice);
	cudaFree(somma);
	cudaFree(res);
	cudaFree(pk);
	cudaFree(trasposta);
	cudaFree(prodotto);
	cudaFree(den);
	cudaFree(res0);
	cudaFree(res1);
	cudaFree(res2);
	cudaFree(red_den);
	cudaFree(scalar);
	cudaFree(matriceB);
	cudaFree(matriceX);
	cudaFreeHost(num);
	cudaFreeHost(deno);

}
