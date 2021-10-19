#include <stdlib.h>
#include <stdio.h>
#include <time.h>


float** trasp(float** matrice,int m,int n){
  int i,j;
  float** res;
  res=(float**)malloc(n*sizeof(float*));
  for(i=0;i<n;i++){
    res[i]=(float*)malloc(m*sizeof(float));
  }
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      res[j][i]=matrice[i][j];
    }
  }
  return res;
}

float** prodottoRXC(float** matriceA,float** matriceB,int mA,int nA,int mB,int nB){
  float** res ;
  int i,j,z;
  float c;
  res=(float**)malloc(mA*sizeof(float*));
  for(i=0;i<mA;i++){
    res[i]=(float*)malloc(nB*sizeof(float));
  }
  for(i=0;i<mA;i++){
    for(j=0;j<nB;j++){
      c=0;
      for(z=0;z<nA;z++){
        c=c+ matriceA[i][z]*matriceB[z][j];
      }
      res[i][j]=c;
    }
  }

  return res;
}

float** sum(float** matriceA,float** matriceB,int m,int n){
  int i,j;
  float** res;
  res=(float**)malloc(m*sizeof(float*));
  for(i=0;i<m;i++){
    res[i]=(float*)malloc(n*sizeof(float));
  }
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      res[i][j]=matriceA[i][j]+matriceB[i][j];
    }
  }
  return res;
}
float** dif(float** matriceA,float** matriceB,int m,int n){
  int i,j;
  float** res;
  res=(float**)malloc(m*sizeof(float*));
  for(i=0;i<m;i++){
    res[i]=(float*)malloc(n*sizeof(float));
  }
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      res[i][j]=matriceA[i][j]-matriceB[i][j];
    }
  }
  return res;
}
float** scalareMatrice(float k,float** matrice,int m,int n){
  int i,j;
  float** res;
  res=(float**)malloc(m*sizeof(float*));
  for(i=0;i<m;i++){
    res[i]=(float*)malloc(n*sizeof(float));
  }
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      res[i][j]=k*matrice[i][j];
    }
  }
  return res;
}
void stampa(float** matrice,int m,int n){
  int i,j;
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      printf("%f  ",matrice[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}
float** inizializzazione(float** matrice,int m,int n){
  int i,j;
  float** res;

  res=(float**)malloc(m*sizeof(float*));
  for(i=0;i<m;i++){
    res[i]=(float*)malloc(n*sizeof(float));
  }
  return res;
}

int main(void) {
  //per prima cosa mi devo calcolare la direzione di discesa
  int N=256;//colonna
  int M=N;//riga
  int NB=1;//colonna
  int MB=N;//riga

  int NX=1;//colonna
  int MX=N;//riga
  int i,j;
  float** matriceA;
  float** matriceB;
  float** matriceX;
  float** pk;
  float** trasposta;
  float** prodotto;
  float** somma;
  float** res;
  float** den;
  float ak;
  srand(time(0));
  res=inizializzazione(res,NX,N);
  den=inizializzazione(den,NX,NX);
  pk=inizializzazione(pk,MB,NB);
  somma=inizializzazione(somma,M,N);
  matriceX=inizializzazione(matriceX,MX,NX);
  matriceA=inizializzazione(matriceA,M,N);
  matriceB=inizializzazione(matriceB,MB,NB);
  //trasposta=inizializzazione(trasposta,B,NB);
  //trasposta=(float**)malloc(NB*sizeof(float*));
  float c=0;
  for(i=0;i<M;i++){
    for(j=0;j<N;j++){
      matriceA[i][j]=c;
      c++;
    }
  }
  c=0;
  for(i=0;i<MB;i++){
    for(j=0;j<NB;j++){
      matriceB[i][j]=c;
      c++;
    }
  }
  c=0;
  for(i=0;i<MX;i++){
    for(j=0;j<NX;j++){
      matriceX[i][j]=c;
      c++;
    }
  }
  //somma=sum(matriceA,matriceB,M,N);
  int f;
  for(f=0;f<1;f++){
    prodotto=prodottoRXC(matriceA,matriceX,M,N,MX,NX);


    pk=dif(matriceB,prodotto,MB,NB);

    trasposta=trasp(pk,M,NX);
    prodotto=prodottoRXC(trasposta,pk,NX,M,M,NX);

    res=prodottoRXC(trasposta,matriceA,NX,M,M,N);


    den=prodottoRXC(res,pk,NX,N,M,NX);
    ak=prodotto[0][0]/den[0][0];
    pk=scalareMatrice(ak,pk,M,NX);
    matriceX=sum(matriceX,pk,MX,NX);
    stampa(matriceX,MB,NB);
  }
  free(pk);
  free(trasposta);
  free(matriceB);
  free(matriceA);
  free(prodotto);

  return 0;
}
