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

__host__ __device__
float4 operator+(const float4 &a, const float4 &b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


__global__
void init_vec(int nels, float* __restrict__ d_vec1)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	d_vec1[i] = i;
}


__global__
void multi_vec2(int nels,int n_row1,int n_col1,int n_row2,int n_col2,float* __restrict__ res_vec,float* __restrict__ d_vec1,float* __restrict__ d_vec2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int r_res,c_res;
	r_res=n_row1;
	c_res=n_row2*n_col2;
	if(i<(r_res*c_res)){
	//int c= blockIdx.x*n_row1 + (threadIdx.x)%n_col1;


	int c= ((int)(i/c_res))*n_row1 + ((int)(i%n_col1))%n_col1;

	int j= ((int)(((int)(i%c_res))/n_row2) + (((int)(i%c_res))%n_row2)*n_col2);

	printf("%f %f\n",d_vec1[c],d_vec2[j] );
	res_vec[i]=d_vec1[c]*d_vec2[j];
	}
}


__global__
void multi_vec(int nels,int n_row1,int n_col1,int n_row2,int n_col2,float* __restrict__ res_vec,
								float* __restrict__ d_vec1,float* __restrict__ d_vec2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int r_res,c_res;
	r_res=n_row1;
	c_res=n_row2*n_col2;
	if(i<(r_res*c_res)){
		int c= ((int)(i/c_res))*n_row1 + ((int)(i%n_col1))%n_col1;
		int j= ((int)(((int)(i%c_res))/n_row2) + (((int)(i%c_res))%n_row2)*n_col2);
		res_vec[i]=d_vec1[c]*d_vec2[j];
	}
}

__global__
void scalareMatrice(float* __restrict__ res_vec,float scalar,float* __restrict__ d_vec)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	res_vec[i]=d_vec[i]*scalar;
}


__global__
void reduction_row(int nels,int l_elem,float* res_vec, float*  d_vec1)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const float4 noels = make_float4(0.0, 0.0, 0.0, 0.0);
	const int nquarts = nels*4;
	const int elem=nels/l_elem;
	int i=idx*l_elem;
	int i0 = i;
	int i1 = i +   4;
	int i2 = i + 2*4;
	int i3 = i + 3*4;
		 __syncthreads();
	float4 r0;
	if(l_elem >= 4){
		r0.x=d_vec1[i];
		r0.y=d_vec1[i+1];
		r0.z=d_vec1[i+2];
		r0.w=d_vec1[i+3];
	}
	else r0= noels;
	float4 r1;
	if(l_elem >= 8){
		r1.x=d_vec1[i1];
		r1.y=d_vec1[i1+1];
		r1.z=d_vec1[i1+2];
		r1.w=d_vec1[i1+3];
	}
	else r1= noels;

	float4 r2;
	if(l_elem >= 12){
		r2.x=d_vec1[i2];
		r2.y=d_vec1[i2+1];
		r2.z=d_vec1[i2+2];
		r2.w=d_vec1[i2+3];
	}
	else r2= noels;
	float4 r3;
	if(l_elem >= 16){
		r3.x=d_vec1[i3];
		r3.y=d_vec1[i3+1];
		r3.z=d_vec1[i3+2];
		r3.w=d_vec1[i3+3];
	}
	else r3= noels;

	float4 v = (r0 + r1) + (r2 + r3);

	if (idx < nels)
		res_vec[idx] = (v.x + v.y) + (v.z + v.w);

}

__global__
void transpose(int nrow,int ncols, float* __restrict__ res_vec, float* __restrict__ d_vec1)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int c =i%ncols;
	int r=i/ncols;
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
  if (argc != 2) {
    error("syntax: vecsum nels v");
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
	err = cudaMalloc((void**)&res0,N*M*N*sizeof(float));
	cuda_check(err, "alloc res0");
	err = cudaMalloc((void**)&prodotto,M*N*N*sizeof(float));
	cuda_check(err, "alloc prodotto");
	err = cudaMalloc((void**)&res1,M*N*sizeof(float));
	cuda_check(err, "alloc res1");
	err = cudaMalloc((void**)&res2,M*N*N*sizeof(float));
	cuda_check(err, "alloc res2");
	err = cudaMalloc((void**)&pk,M*N*sizeof(float));
	cuda_check(err, "alloc pk");
	err = cudaMalloc((void**)&trasposta,M*N*sizeof(float));
	cuda_check(err, "alloc trasposta	");
	err = cudaMalloc((void**)&den,M*N*sizeof(float));
	cuda_check(err, "alloc den");
	err = cudaMalloc((void**)&red_den,M*N*sizeof(float));
	cuda_check(err, "alloc den");
	err = cudaMalloc((void**)&scalar,M*N*sizeof(float));
	cuda_check(err, "alloc scalar");

	cudaEvent_t pre_init, post_init, pre_sum, post_sum,pre_prodotto,post_prodotto,
							pre_transpose,post_transpose,pre_scalar_matrice,post_scalar_matrice,pre_vecsum,post_vecsum,
							pre_vecdif,post_vecdif;


	err = cudaEventCreate(&pre_init, 0);
	cuda_check(err, "create pre_init");
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
  const int blockSize = 512;
	int numBlocks = (nels + blockSize - 1)/blockSize;


	cudaEventRecord(pre_init);
	init_vec<<<blockSize,numBlocks>>>(nels, matriceA);
	cudaEventRecord(post_init);

	numBlocks = (M*N + blockSize - 1)/blockSize;
  init_vec<<<blockSize, numBlocks>>>(M*N, matriceB);

	init_vec<<<blockSize, numBlocks>>>(M*N, matriceX);
	int i;

	//calcolo i parametri della riduzione
	int THREAD_LOAD=0;
	float n = N;
	while (n > 1) {
		n/=4;
		if(n==1){
			THREAD_LOAD=4;
		}
	}
	n = N;
	while (n > 1) {

		n/=8;
		if(n==1){
			THREAD_LOAD=8;
		}
	}
	n=N;
	while (n > 1) {
		n/=12;
		if(n==1){
			THREAD_LOAD=12;
		}
	}
	while (n > 1) {
		n/=16;
		if(n==1){
			THREAD_LOAD=16;
		}
	}
	if(THREAD_LOAD==0){
		printf("Errore N deve essere una potenza di 4,8,12,16");
		exit(0);
	}
	int j;
	int c=N;
	float* temp;
	float runtime_red_ms;
	int lr=0;
	int log=N*N;
	while(log>N){
		++lr;
		log=log/THREAD_LOAD;
	}

	cudaEvent_t pre_red[lr], post_red[lr];
	//inizializzo gli eventi per la riduzione
	for(i=0;i<lr;i++){
		err = cudaEventCreate(&(pre_red[i]), 0);
		cuda_check(err, "create pre_red");

		err = cudaEventCreate(&(post_red[i]), 0);
		cuda_check(err, "create post_red");
	}
	for(i=0;i<1;i++){
		numBlocks = (nels + blockSize - 1)/blockSize;
		cudaEventRecord(pre_prodotto);
		multi_vec<<<blockSize, numBlocks>>>(nels*M,N,N,N,M,somma,matriceA,matriceX);
		cudaEventRecord(post_prodotto);

		c=N*N;

		int nels_red=0;
		int cont=0;
		while(c>N){
			c/=THREAD_LOAD;
			nels_red+=c;

			numBlocks = (c + blockSize - 1)/blockSize;
			cudaEventRecord(pre_red[cont]);
			reduction_row<<<blockSize, numBlocks>>>(c,THREAD_LOAD,res0,somma);
			cudaEventRecord(post_red[cont]);
			err = cudaMemcpy(somma, res0, c*sizeof(float), cudaMemcpyDeviceToDevice);
			cuda_check(err, "cpy");
			cont++;
		}

		printf("%d %d\n",lr,nels_red );
		numBlocks = (N*M + blockSize - 1)/blockSize;
		cudaEventRecord(pre_vecdif);
		vecdif<<<blockSize, numBlocks>>>(N*M,pk,matriceB,res0);
		cudaEventRecord(post_vecdif);

		numBlocks = (N*N + blockSize - 1)/blockSize;
		cudaEventRecord(pre_transpose);
		transpose<<<blockSize, numBlocks>>>(N,M,trasposta,pk);
		cudaEventRecord(post_transpose);
		numBlocks = (M*N + blockSize - 1)/blockSize;
		multi_vec<<<blockSize, numBlocks>>>(N*M,M,N,N,M,prodotto,trasposta,pk);
		c=N;
		while (c>1) {
			c/=THREAD_LOAD;
			numBlocks = (c + blockSize - 1)/blockSize;
			reduction_row<<<blockSize, numBlocks>>>(c,THREAD_LOAD,res1,prodotto);
			err = cudaMemcpy(prodotto, res1, c*sizeof(float), cudaMemcpyDeviceToDevice);
			cuda_check(err, "cpy");
		}

		numBlocks = (M*N*N*M + blockSize - 1)/blockSize;
		multi_vec<<<blockSize, numBlocks>>>(M*N*N*M,M,N,N,N,res,trasposta,matriceA);

		c=N*N;
		while (c>N) {
			c/=THREAD_LOAD;
			numBlocks = (c + blockSize - 1)/blockSize;
			reduction_row<<<blockSize, numBlocks>>>(c,THREAD_LOAD,res2,res);
			err = cudaMemcpy(res, res2, c*sizeof(float), cudaMemcpyDeviceToDevice);
			cuda_check(err, "cpy");
		}


		numBlocks = (N*N + blockSize - 1)/blockSize;
		multi_vec<<<blockSize, numBlocks>>>(N*N,M,N,N,M,den,res2,pk);
		c=N;
		while (c>1) {
			c/=THREAD_LOAD;
			numBlocks = (c + blockSize - 1)/blockSize;
			reduction_row<<<blockSize, numBlocks>>>(c,THREAD_LOAD,red_den,den);
			err = cudaMemcpy(den, red_den, c*sizeof(float), cudaMemcpyDeviceToDevice);
			cuda_check(err, "cpy");
		}
		err = cudaMemcpy(num, res1, 1*sizeof(float), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(deno, red_den, 1*sizeof(float), cudaMemcpyDeviceToHost);
		ak=num[0]/deno[0];
		printf("%f\n",ak );
		numBlocks = (N + blockSize - 1)/blockSize;
		cudaEventRecord(pre_scalar_matrice);
		scalareMatrice<<<blockSize, numBlocks>>>(scalar,ak,pk);
		cudaEventRecord(post_scalar_matrice);

		numBlocks = (N*N + blockSize - 1)/blockSize;
		cudaEventRecord(pre_vecsum);
		vecsum<<<blockSize, numBlocks>>>(N*M*N,matriceX,matriceX,scalar);
		cudaEventRecord(post_vecsum);

		err = cudaMemcpy(matrice, matriceX, M*N*sizeof(float), cudaMemcpyDeviceToHost);
		cuda_check(err, "create mem");
	  stampa(matrice,M*N);



		float runtime_init_ms, runtime_prodotto_ms, runtime_red_ms,runtime_transpose_ms,runtime_scalar_matrice_ms,
					runtime_vecdif_ms,runtime_vecsum_ms,runtime_red_count_ms;
		err = cudaEventElapsedTime(&runtime_init_ms, pre_init, post_init);
		cuda_check(err, "elapsed time init");
		err = cudaEventElapsedTime(&runtime_prodotto_ms, pre_prodotto, post_prodotto);
		cuda_check(err, "elapsed time prodotto");
		runtime_red_count_ms=0;
		for(j=0;j<lr;j++){

			err = cudaEventElapsedTime(&runtime_red_ms, pre_red[j], post_red[j]);
			cuda_check(err, "elapsed time reduction");
			runtime_red_count_ms+=runtime_red_ms;

		}
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
			runtime_red_count_ms, nels_red/runtime_red_count_ms/1.0e6, (nels_red*sizeof(float))/runtime_red_count_ms/1.0e6);
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
