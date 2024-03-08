#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

void canny() {
	
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	uint8_t *im; int height; int width;
	uint8_t *imEdge; float *NR; float *G; float *phi; float *Gx; float *Gy;
	uint8_t *pedge;
	float *sin_table; float *cos_table; 
	uint32_t *accum; int accu_height; int accu_width;
	int *x1; int *y1; int *x2; int *y2; int *nlines;
	int threshold;
	
	/* Canny */
	canny(im, imEdge,
		NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);

	/* hough transform */
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table);

	if (width>height) threshold = width/6;
	else threshold = height/6;


	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}
