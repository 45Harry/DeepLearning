��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028Ԕ
n
ConstConst*
_output_shapes

:*
dtype0*1
value(B&"c$�@R�]>���I�o�@�͑@��@
p
Const_1Const*
_output_shapes

:*
dtype0*1
value(B&"��@���? 	�D^=@�B2+��
l
Const_2Const*
_output_shapes

:*
dtype0*-
value$B""�Li@��Cc$�@R�]>���I
l
Const_3Const*
_output_shapes

:*
dtype0*-
value$B""�y@��A��@���? 	�D
�
my_cool_model/biasVarHandleOp*
_output_shapes
: *#

debug_namemy_cool_model/bias/*
dtype0*
shape:*#
shared_namemy_cool_model/bias
u
&my_cool_model/bias/Read/ReadVariableOpReadVariableOpmy_cool_model/bias*
_output_shapes
:*
dtype0
�
my_cool_model/kernelVarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/kernel/*
dtype0*
shape
:*%
shared_namemy_cool_model/kernel
}
(my_cool_model/kernel/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel*
_output_shapes

:*
dtype0
�
my_cool_model/bias_1VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_1/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_1
y
(my_cool_model/bias_1/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_1*
_output_shapes
:*
dtype0
�
my_cool_model/bias_2VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_2/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_2
y
(my_cool_model/bias_2/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_2*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_1VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_1/*
dtype0*
shape
:#*'
shared_namemy_cool_model/kernel_1
�
*my_cool_model/kernel_1/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_1*
_output_shapes

:#*
dtype0
�
my_cool_model/bias_3VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_3/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_3
y
(my_cool_model/bias_3/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_3*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_2VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_2/*
dtype0*
shape
:*'
shared_namemy_cool_model/kernel_2
�
*my_cool_model/kernel_2/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_2*
_output_shapes

:*
dtype0
�
my_cool_model/kernel_3VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_3/*
dtype0*
shape
:*'
shared_namemy_cool_model/kernel_3
�
*my_cool_model/kernel_3/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_3*
_output_shapes

:*
dtype0
�
my_cool_model/bias_4VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_4/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_4
y
(my_cool_model/bias_4/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_4*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpmy_cool_model/bias_4*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_4VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_4/*
dtype0*
shape
:*'
shared_namemy_cool_model/kernel_4
�
*my_cool_model/kernel_4/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_4*
_output_shapes

:*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpmy_cool_model/kernel_4*
_class
loc:@Variable_1*
_output_shapes

:*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:*
dtype0
�
my_cool_model/bias_5VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_5/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_5
y
(my_cool_model/bias_5/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_5*
_output_shapes
:*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpmy_cool_model/bias_5*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_5VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_5/*
dtype0*
shape
:#*'
shared_namemy_cool_model/kernel_5
�
*my_cool_model/kernel_5/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_5*
_output_shapes

:#*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpmy_cool_model/kernel_5*
_class
loc:@Variable_3*
_output_shapes

:#*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape
:#*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:#*
dtype0
�
my_cool_model/bias_6VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_6/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_6
y
(my_cool_model/bias_6/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_6*
_output_shapes
:*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpmy_cool_model/bias_6*
_class
loc:@Variable_4*
_output_shapes
:*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_6VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_6/*
dtype0*
shape
:*'
shared_namemy_cool_model/kernel_6
�
*my_cool_model/kernel_6/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_6*
_output_shapes

:*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpmy_cool_model/kernel_6*
_class
loc:@Variable_5*
_output_shapes

:*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape
:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:*
dtype0
�
my_cool_model/bias_7VarHandleOp*
_output_shapes
: *%

debug_namemy_cool_model/bias_7/*
dtype0*
shape:*%
shared_namemy_cool_model/bias_7
y
(my_cool_model/bias_7/Read/ReadVariableOpReadVariableOpmy_cool_model/bias_7*
_output_shapes
:*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpmy_cool_model/bias_7*
_class
loc:@Variable_6*
_output_shapes
:*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:*
dtype0
�
my_cool_model/kernel_7VarHandleOp*
_output_shapes
: *'

debug_namemy_cool_model/kernel_7/*
dtype0*
shape
:*'
shared_namemy_cool_model/kernel_7
�
*my_cool_model/kernel_7/Read/ReadVariableOpReadVariableOpmy_cool_model/kernel_7*
_output_shapes

:*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpmy_cool_model/kernel_7*
_class
loc:@Variable_7*
_output_shapes

:*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape
:*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
i
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes

:*
dtype0
�
normalization_3/countVarHandleOp*
_output_shapes
: *&

debug_namenormalization_3/count/*
dtype0	*
shape: *&
shared_namenormalization_3/count
w
)normalization_3/count/Read/ReadVariableOpReadVariableOpnormalization_3/count*
_output_shapes
: *
dtype0	
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpnormalization_3/count*
_class
loc:@Variable_8*
_output_shapes
: *
dtype0	
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0	*
shape: *
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0	
a
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0	
�
normalization_3/varianceVarHandleOp*
_output_shapes
: *)

debug_namenormalization_3/variance/*
dtype0*
shape:*)
shared_namenormalization_3/variance
�
,normalization_3/variance/Read/ReadVariableOpReadVariableOpnormalization_3/variance*
_output_shapes
:*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpnormalization_3/variance*
_class
loc:@Variable_9*
_output_shapes
:*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:*
dtype0
�
normalization_3/meanVarHandleOp*
_output_shapes
: *%

debug_namenormalization_3/mean/*
dtype0*
shape:*%
shared_namenormalization_3/mean
y
(normalization_3/mean/Read/ReadVariableOpReadVariableOpnormalization_3/mean*
_output_shapes
:*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpnormalization_3/mean*
_class
loc:@Variable_10*
_output_shapes
:*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
g
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:*
dtype0
�
normalization_2/countVarHandleOp*
_output_shapes
: *&

debug_namenormalization_2/count/*
dtype0	*
shape: *&
shared_namenormalization_2/count
w
)normalization_2/count/Read/ReadVariableOpReadVariableOpnormalization_2/count*
_output_shapes
: *
dtype0	
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpnormalization_2/count*
_class
loc:@Variable_11*
_output_shapes
: *
dtype0	
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0	*
shape: *
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0	
c
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
: *
dtype0	
�
normalization_2/varianceVarHandleOp*
_output_shapes
: *)

debug_namenormalization_2/variance/*
dtype0*
shape:*)
shared_namenormalization_2/variance
�
,normalization_2/variance/Read/ReadVariableOpReadVariableOpnormalization_2/variance*
_output_shapes
:*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpnormalization_2/variance*
_class
loc:@Variable_12*
_output_shapes
:*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:*
dtype0
�
normalization_2/meanVarHandleOp*
_output_shapes
: *%

debug_namenormalization_2/mean/*
dtype0*
shape:*%
shared_namenormalization_2/mean
y
(normalization_2/mean/Read/ReadVariableOpReadVariableOpnormalization_2/mean*
_output_shapes
:*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpnormalization_2/mean*
_class
loc:@Variable_13*
_output_shapes
:*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
g
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:*
dtype0
o
serve_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
q
serve_args_0_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserve_args_0serve_args_0_1Const_3Const_2Const_1Constmy_cool_model/kernel_7my_cool_model/bias_7my_cool_model/kernel_6my_cool_model/bias_6my_cool_model/kernel_5my_cool_model/bias_5my_cool_model/kernel_4my_cool_model/bias_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *6
f1R/
-__inference_signature_wrapper___call___505997
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_args_0_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_args_0serving_default_args_0_1Const_3Const_2Const_1Constmy_cool_model/kernel_7my_cool_model/bias_7my_cool_model/kernel_6my_cool_model/bias_6my_cool_model/kernel_5my_cool_model/bias_5my_cool_model/kernel_4my_cool_model/bias_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *6
f1R/
-__inference_signature_wrapper___call___506029

NoOpNoOp
�
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
j
0
	1

2
3
4
5
6
7
8
9
10
11
12
13*
<
0
1
2
3
4
5
6
7*
.
0
	1

2
3
4
5*
<
0
1
2
3
4
5
6
7*
* 

trace_0* 
"
	serve
 serving_default* 
KE
VARIABLE_VALUEVariable_13&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_12&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_11&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_10&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_9&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_8&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_7&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_6&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_5&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_4&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmy_cool_model/kernel_6+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmy_cool_model/kernel_7+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmy_cool_model/bias_7+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmy_cool_model/kernel_5+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmy_cool_model/bias_6+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmy_cool_model/bias_5+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmy_cool_model/kernel_4+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmy_cool_model/bias_4+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
>
!	capture_0
"	capture_1
#	capture_2
$	capture_3* 
>
!	capture_0
"	capture_1
#	capture_2
$	capture_3* 
>
!	capture_0
"	capture_1
#	capture_2
$	capture_3* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablemy_cool_model/kernel_6my_cool_model/kernel_7my_cool_model/bias_7my_cool_model/kernel_5my_cool_model/bias_6my_cool_model/bias_5my_cool_model/kernel_4my_cool_model/bias_4Const_4*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_save_506249
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablemy_cool_model/kernel_6my_cool_model/kernel_7my_cool_model/bias_7my_cool_model/kernel_5my_cool_model/bias_6my_cool_model/bias_5my_cool_model/kernel_4my_cool_model/bias_4*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *+
f&R$
"__inference__traced_restore_506324��
��
�
__inference__traced_save_506249
file_prefix0
"read_disablecopyonread_variable_13:2
$read_1_disablecopyonread_variable_12:.
$read_2_disablecopyonread_variable_11:	 2
$read_3_disablecopyonread_variable_10:1
#read_4_disablecopyonread_variable_9:-
#read_5_disablecopyonread_variable_8:	 5
#read_6_disablecopyonread_variable_7:1
#read_7_disablecopyonread_variable_6:5
#read_8_disablecopyonread_variable_5:1
#read_9_disablecopyonread_variable_4:6
$read_10_disablecopyonread_variable_3:#2
$read_11_disablecopyonread_variable_2:6
$read_12_disablecopyonread_variable_1:0
"read_13_disablecopyonread_variable:B
0read_14_disablecopyonread_my_cool_model_kernel_6:B
0read_15_disablecopyonread_my_cool_model_kernel_7:<
.read_16_disablecopyonread_my_cool_model_bias_7:B
0read_17_disablecopyonread_my_cool_model_kernel_5:#<
.read_18_disablecopyonread_my_cool_model_bias_6:<
.read_19_disablecopyonread_my_cool_model_bias_5:B
0read_20_disablecopyonread_my_cool_model_kernel_4:<
.read_21_disablecopyonread_my_cool_model_bias_4:
savev2_const_4
identity_45��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_13*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_13^Read/DisableCopyOnRead*
_output_shapes
:*
dtype0V
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_12*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_12^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_11*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_11^Read_2/DisableCopyOnRead*
_output_shapes
: *
dtype0	V

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_10*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_10^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_9*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_9^Read_4/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_8*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_8^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0	W
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0	*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
: h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_7*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_7^Read_6/DisableCopyOnRead*
_output_shapes

:*
dtype0_
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:h
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_variable_6*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_variable_6^Read_7/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_variable_5*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_variable_5^Read_8/DisableCopyOnRead*
_output_shapes

:*
dtype0_
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:h
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_variable_4*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_variable_4^Read_9/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_variable_3*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_variable_3^Read_10/DisableCopyOnRead*
_output_shapes

:#*
dtype0`
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes

:#e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:#j
Read_11/DisableCopyOnReadDisableCopyOnRead$read_11_disablecopyonread_variable_2*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp$read_11_disablecopyonread_variable_2^Read_11/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_12/DisableCopyOnReadDisableCopyOnRead$read_12_disablecopyonread_variable_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp$read_12_disablecopyonread_variable_1^Read_12/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:h
Read_13/DisableCopyOnReadDisableCopyOnRead"read_13_disablecopyonread_variable*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp"read_13_disablecopyonread_variable^Read_13/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_my_cool_model_kernel_6*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_my_cool_model_kernel_6^Read_14/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:v
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_my_cool_model_kernel_7*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_my_cool_model_kernel_7^Read_15/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:t
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_my_cool_model_bias_7*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_my_cool_model_bias_7^Read_16/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_my_cool_model_kernel_5*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_my_cool_model_kernel_5^Read_17/DisableCopyOnRead*
_output_shapes

:#*
dtype0`
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes

:#e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:#t
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_my_cool_model_bias_6*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_my_cool_model_bias_6^Read_18/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_my_cool_model_bias_5*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_my_cool_model_bias_5^Read_19/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_my_cool_model_kernel_4*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_my_cool_model_kernel_4^Read_20/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:t
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_my_cool_model_bias_4*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_my_cool_model_bias_4^Read_21/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_4:40
.
_user_specified_namemy_cool_model/bias_4:62
0
_user_specified_namemy_cool_model/kernel_4:40
.
_user_specified_namemy_cool_model/bias_5:40
.
_user_specified_namemy_cool_model/bias_6:62
0
_user_specified_namemy_cool_model/kernel_5:40
.
_user_specified_namemy_cool_model/bias_7:62
0
_user_specified_namemy_cool_model/kernel_7:62
0
_user_specified_namemy_cool_model/kernel_6:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*
&
$
_user_specified_name
Variable_4:*	&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�d
�
"__inference__traced_restore_506324
file_prefix*
assignvariableop_variable_13:,
assignvariableop_1_variable_12:(
assignvariableop_2_variable_11:	 ,
assignvariableop_3_variable_10:+
assignvariableop_4_variable_9:'
assignvariableop_5_variable_8:	 /
assignvariableop_6_variable_7:+
assignvariableop_7_variable_6:/
assignvariableop_8_variable_5:+
assignvariableop_9_variable_4:0
assignvariableop_10_variable_3:#,
assignvariableop_11_variable_2:0
assignvariableop_12_variable_1:*
assignvariableop_13_variable:<
*assignvariableop_14_my_cool_model_kernel_6:<
*assignvariableop_15_my_cool_model_kernel_7:6
(assignvariableop_16_my_cool_model_bias_7:<
*assignvariableop_17_my_cool_model_kernel_5:#6
(assignvariableop_18_my_cool_model_bias_6:6
(assignvariableop_19_my_cool_model_bias_5:<
*assignvariableop_20_my_cool_model_kernel_4:6
(assignvariableop_21_my_cool_model_bias_4:
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_13Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_12Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_11Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_10Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_9Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_8Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_7Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_6Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_5Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_4Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_3Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_2Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variableIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_my_cool_model_kernel_6Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_my_cool_model_kernel_7Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_my_cool_model_bias_7Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_my_cool_model_kernel_5Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_my_cool_model_bias_6Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_my_cool_model_bias_5Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_my_cool_model_kernel_4Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_my_cool_model_bias_4Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:40
.
_user_specified_namemy_cool_model/bias_4:62
0
_user_specified_namemy_cool_model/kernel_4:40
.
_user_specified_namemy_cool_model/bias_5:40
.
_user_specified_namemy_cool_model/bias_6:62
0
_user_specified_namemy_cool_model/kernel_5:40
.
_user_specified_namemy_cool_model/bias_7:62
0
_user_specified_namemy_cool_model/kernel_7:62
0
_user_specified_namemy_cool_model/kernel_6:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*
&
$
_user_specified_name
Variable_4:*	&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_signature_wrapper___call___506029

args_0
args_0_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___505964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:���������::::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name506023:&"
 
_user_specified_name506021:&"
 
_user_specified_name506019:&
"
 
_user_specified_name506017:&	"
 
_user_specified_name506015:&"
 
_user_specified_name506013:&"
 
_user_specified_name506011:&"
 
_user_specified_name506009:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::QM
'
_output_shapes
:���������
"
_user_specified_name
args_0_1:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0
�
�
-__inference_signature_wrapper___call___505997

args_0
args_0_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___505964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:���������::::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name505991:&"
 
_user_specified_name505989:&"
 
_user_specified_name505987:&
"
 
_user_specified_name505985:&	"
 
_user_specified_name505983:&"
 
_user_specified_name505981:&"
 
_user_specified_name505979:&"
 
_user_specified_name505977:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::QM
'
_output_shapes
:���������
"
_user_specified_name
args_0_1:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0
�C
�	
__inference___call___505964

args_0
args_0_1+
'my_cool_model_1_normalization_2_1_sub_y,
(my_cool_model_1_normalization_2_1_sqrt_x+
'my_cool_model_1_normalization_3_1_sub_y,
(my_cool_model_1_normalization_3_1_sqrt_xH
6my_cool_model_1_dense_4_1_cast_readvariableop_resource:G
9my_cool_model_1_dense_4_1_biasadd_readvariableop_resource:H
6my_cool_model_1_dense_5_1_cast_readvariableop_resource:G
9my_cool_model_1_dense_5_1_biasadd_readvariableop_resource:H
6my_cool_model_1_dense_6_1_cast_readvariableop_resource:#C
5my_cool_model_1_dense_6_1_add_readvariableop_resource:H
6my_cool_model_1_dense_7_1_cast_readvariableop_resource:C
5my_cool_model_1_dense_7_1_add_readvariableop_resource:
identity

identity_1��0my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOp�-my_cool_model_1/dense_4_1/Cast/ReadVariableOp�0my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOp�-my_cool_model_1/dense_5_1/Cast/ReadVariableOp�,my_cool_model_1/dense_6_1/Add/ReadVariableOp�-my_cool_model_1/dense_6_1/Cast/ReadVariableOp�,my_cool_model_1/dense_7_1/Add/ReadVariableOp�-my_cool_model_1/dense_7_1/Cast/ReadVariableOp�
%my_cool_model_1/normalization_2_1/SubSubargs_0'my_cool_model_1_normalization_2_1_sub_y*
T0*'
_output_shapes
:����������
&my_cool_model_1/normalization_2_1/SqrtSqrt(my_cool_model_1_normalization_2_1_sqrt_x*
T0*
_output_shapes

:l
'my_cool_model_1/normalization_2_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
)my_cool_model_1/normalization_2_1/MaximumMaximum*my_cool_model_1/normalization_2_1/Sqrt:y:00my_cool_model_1/normalization_2_1/Const:output:0*
T0*
_output_shapes

:�
)my_cool_model_1/normalization_2_1/truedivRealDiv)my_cool_model_1/normalization_2_1/Sub:z:0-my_cool_model_1/normalization_2_1/Maximum:z:0*
T0*'
_output_shapes
:����������
%my_cool_model_1/normalization_3_1/SubSubargs_0_1'my_cool_model_1_normalization_3_1_sub_y*
T0*'
_output_shapes
:����������
&my_cool_model_1/normalization_3_1/SqrtSqrt(my_cool_model_1_normalization_3_1_sqrt_x*
T0*
_output_shapes

:l
'my_cool_model_1/normalization_3_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
)my_cool_model_1/normalization_3_1/MaximumMaximum*my_cool_model_1/normalization_3_1/Sqrt:y:00my_cool_model_1/normalization_3_1/Const:output:0*
T0*
_output_shapes

:�
)my_cool_model_1/normalization_3_1/truedivRealDiv)my_cool_model_1/normalization_3_1/Sub:z:0-my_cool_model_1/normalization_3_1/Maximum:z:0*
T0*'
_output_shapes
:����������
-my_cool_model_1/dense_4_1/Cast/ReadVariableOpReadVariableOp6my_cool_model_1_dense_4_1_cast_readvariableop_resource*
_output_shapes

:*
dtype0�
 my_cool_model_1/dense_4_1/MatMulMatMul-my_cool_model_1/normalization_3_1/truediv:z:05my_cool_model_1/dense_4_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOpReadVariableOp9my_cool_model_1_dense_4_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!my_cool_model_1/dense_4_1/BiasAddBiasAdd*my_cool_model_1/dense_4_1/MatMul:product:08my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
my_cool_model_1/dense_4_1/ReluRelu*my_cool_model_1/dense_4_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-my_cool_model_1/dense_5_1/Cast/ReadVariableOpReadVariableOp6my_cool_model_1_dense_5_1_cast_readvariableop_resource*
_output_shapes

:*
dtype0�
 my_cool_model_1/dense_5_1/MatMulMatMul,my_cool_model_1/dense_4_1/Relu:activations:05my_cool_model_1/dense_5_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOpReadVariableOp9my_cool_model_1_dense_5_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!my_cool_model_1/dense_5_1/BiasAddBiasAdd*my_cool_model_1/dense_5_1/MatMul:product:08my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
my_cool_model_1/dense_5_1/ReluRelu*my_cool_model_1/dense_5_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
+my_cool_model_1/concatenate_7_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
&my_cool_model_1/concatenate_7_1/concatConcatV2-my_cool_model_1/normalization_2_1/truediv:z:0,my_cool_model_1/dense_5_1/Relu:activations:04my_cool_model_1/concatenate_7_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������#�
-my_cool_model_1/dense_6_1/Cast/ReadVariableOpReadVariableOp6my_cool_model_1_dense_6_1_cast_readvariableop_resource*
_output_shapes

:#*
dtype0�
 my_cool_model_1/dense_6_1/MatMulMatMul/my_cool_model_1/concatenate_7_1/concat:output:05my_cool_model_1/dense_6_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,my_cool_model_1/dense_6_1/Add/ReadVariableOpReadVariableOp5my_cool_model_1_dense_6_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
my_cool_model_1/dense_6_1/AddAddV2*my_cool_model_1/dense_6_1/MatMul:product:04my_cool_model_1/dense_6_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-my_cool_model_1/dense_7_1/Cast/ReadVariableOpReadVariableOp6my_cool_model_1_dense_7_1_cast_readvariableop_resource*
_output_shapes

:*
dtype0�
 my_cool_model_1/dense_7_1/MatMulMatMul,my_cool_model_1/dense_5_1/Relu:activations:05my_cool_model_1/dense_7_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,my_cool_model_1/dense_7_1/Add/ReadVariableOpReadVariableOp5my_cool_model_1_dense_7_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
my_cool_model_1/dense_7_1/AddAddV2*my_cool_model_1/dense_7_1/MatMul:product:04my_cool_model_1/dense_7_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!my_cool_model_1/dense_6_1/Add:z:0^NoOp*
T0*'
_output_shapes
:���������r

Identity_1Identity!my_cool_model_1/dense_7_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOp.^my_cool_model_1/dense_4_1/Cast/ReadVariableOp1^my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOp.^my_cool_model_1/dense_5_1/Cast/ReadVariableOp-^my_cool_model_1/dense_6_1/Add/ReadVariableOp.^my_cool_model_1/dense_6_1/Cast/ReadVariableOp-^my_cool_model_1/dense_7_1/Add/ReadVariableOp.^my_cool_model_1/dense_7_1/Cast/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:���������::::: : : : : : : : 2d
0my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOp0my_cool_model_1/dense_4_1/BiasAdd/ReadVariableOp2^
-my_cool_model_1/dense_4_1/Cast/ReadVariableOp-my_cool_model_1/dense_4_1/Cast/ReadVariableOp2d
0my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOp0my_cool_model_1/dense_5_1/BiasAdd/ReadVariableOp2^
-my_cool_model_1/dense_5_1/Cast/ReadVariableOp-my_cool_model_1/dense_5_1/Cast/ReadVariableOp2\
,my_cool_model_1/dense_6_1/Add/ReadVariableOp,my_cool_model_1/dense_6_1/Add/ReadVariableOp2^
-my_cool_model_1/dense_6_1/Cast/ReadVariableOp-my_cool_model_1/dense_6_1/Cast/ReadVariableOp2\
,my_cool_model_1/dense_7_1/Add/ReadVariableOp,my_cool_model_1/dense_7_1/Add/ReadVariableOp2^
-my_cool_model_1/dense_7_1/Cast/ReadVariableOp-my_cool_model_1/dense_7_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::OK
'
_output_shapes
:���������
 
_user_specified_nameargs_0:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
3
args_0_1'
serve_args_0_1:0���������
/
args_0%
serve_args_0:0���������<
output_00
StatefulPartitionedCall:0���������<
output_10
StatefulPartitionedCall:1���������tensorflow/serving/predict*�
serving_default�
=
args_0_11
serving_default_args_0_1:0���������
9
args_0/
serving_default_args_0:0���������>
output_02
StatefulPartitionedCall_1:0���������>
output_12
StatefulPartitionedCall_1:1���������tensorflow/serving/predict:�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_02�
__inference___call___505964�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *<�9
7�4
����������
����������ztrace_0
7
	serve
 serving_default"
signature_map
 :2normalization_2/mean
$:"2normalization_2/variance
:	 2normalization_2/count
 :2normalization_3/mean
$:"2normalization_3/variance
:	 2normalization_3/count
&:$2my_cool_model/kernel
 :2my_cool_model/bias
&:$2my_cool_model/kernel
 :2my_cool_model/bias
&:$#2my_cool_model/kernel
 :2my_cool_model/bias
&:$2my_cool_model/kernel
 :2my_cool_model/bias
&:$2my_cool_model/kernel
&:$2my_cool_model/kernel
 :2my_cool_model/bias
&:$#2my_cool_model/kernel
 :2my_cool_model/bias
 :2my_cool_model/bias
&:$2my_cool_model/kernel
 :2my_cool_model/bias
�
!	capture_0
"	capture_1
#	capture_2
$	capture_3B�
__inference___call___505964args_0args_0_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z!	capture_0z"	capture_1z#	capture_2z$	capture_3
�
!	capture_0
"	capture_1
#	capture_2
$	capture_3B�
-__inference_signature_wrapper___call___505997args_0args_0_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 '

kwonlyargs�
jargs_0

jargs_0_1
kwonlydefaults
 
annotations� *
 z!	capture_0z"	capture_1z#	capture_2z$	capture_3
�
!	capture_0
"	capture_1
#	capture_2
$	capture_3B�
-__inference_signature_wrapper___call___506029args_0args_0_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 '

kwonlyargs�
jargs_0

jargs_0_1
kwonlydefaults
 
annotations� *
 z!	capture_0z"	capture_1z#	capture_2z$	capture_3
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant�
__inference___call___505964�!"#$Z�W
P�M
K�H
"�
args_0_0���������
"�
args_0_1���������
� "K�H
"�
tensor_0���������
"�
tensor_1����������
-__inference_signature_wrapper___call___505997�!"#$i�f
� 
_�\
.
args_0_1"�
args_0_1���������
*
args_0 �
args_0���������"c�`
.
output_0"�
output_0���������
.
output_1"�
output_1����������
-__inference_signature_wrapper___call___506029�!"#$i�f
� 
_�\
.
args_0_1"�
args_0_1���������
*
args_0 �
args_0���������"c�`
.
output_0"�
output_0���������
.
output_1"�
output_1���������