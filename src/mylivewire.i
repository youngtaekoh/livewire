/* mylivewire.i */
%module mylivewire
%{
#define SWIG_FILE_WITH_INIT
#include "mylivewire.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *path, int row, int col)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *gradImg, int row2, int col2)}
%typemap(in) int[2](int temp[2]) {   // temp[4] becomes a local variable
	int i;
	if (PyTuple_Check($input)) {
		if (!PyArg_ParseTuple($input,"ii",temp,temp+1)) {
			PyErr_SetString(PyExc_TypeError,"tuple must have 2 elements");
			return NULL;
		}
		$1 = &temp[0];
	} else {
		PyErr_SetString(PyExc_TypeError,"expected a tuple.");
		return NULL;
	}
}

%include "mylivewire.h"
