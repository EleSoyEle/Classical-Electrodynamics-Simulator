/*

Todo calculo sobre las particulas será en python.
En C solo se va a calcular la dinamica de los campos


--------Requerimientos para este codigo--------

1. La densidad de corriente y carga será solamente dada como mallas para evitar personalizar
en un numero especifico de particulas
2. Cuidar desbordamiento

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Modificar la k en caso de haber desbordamiento en la delta de Dirac
#define k  0.5
#define PI 3.14159

//Crea una malla de espacio en 1D en un intervalo [a,b]
double* Make1DSpace(double a,double b,int size){
    double* space = (double*)calloc(size,sizeof(double));
    double dx = (b-a)/(double)size;
    for(int i=0;i<size;i++){
        space[i] = a + dx*i;
    }
    return space;
}

double* DiracDelta1D(double x, double x0,double* space,double a, double b, int size){
    double dx = (b-a)/(double)size;
    double std = dx/k;
    double* delta = (double*)calloc(size,sizeof(double));

    double div_term = std*sqrt(2*PI);
    double exp_div_term = 2*pow(std, 2.0);
    for(int i = 0;i<size;i++){
        delta[i] = exp(-pow(space[i],2.0)/exp_div_term)/div_term;
    }
    return delta;    
}

double* Make2DSpace(double x_range[2],double y_range[2], int size){
    double* x_line = Make1DSpace(x_range[0],y_range[0],size);
    double* y_line = Make1DSpace(x_range[1],y_range[1],size);
    double* space = (double*)calloc(size*size*2,sizeof(double));

    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            space[i*size+j*size+0] = x_line[i];
            space[i*size+j*size+1] = y_line[i];
        }
    }
    free(x_line);
    free(y_line);
    return space;
}


int main(){
    return 0;
}