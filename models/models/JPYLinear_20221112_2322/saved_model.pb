Α

αΔ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8ηκ
’
%Adam/jpy_model_linaer/output_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/jpy_model_linaer/output_1/bias/v

9Adam/jpy_model_linaer/output_1/bias/v/Read/ReadVariableOpReadVariableOp%Adam/jpy_model_linaer/output_1/bias/v*
_output_shapes
:*
dtype0
«
'Adam/jpy_model_linaer/output_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*8
shared_name)'Adam/jpy_model_linaer/output_1/kernel/v
€
;Adam/jpy_model_linaer/output_1/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/jpy_model_linaer/output_1/kernel/v*
_output_shapes
:	*
dtype0
©
(Adam/jpy_model_linaer/fifth_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/fifth_dense/bias/v
’
<Adam/jpy_model_linaer/fifth_dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/fifth_dense/bias/v*
_output_shapes	
:*
dtype0
²
*Adam/jpy_model_linaer/fifth_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/jpy_model_linaer/fifth_dense/kernel/v
«
>Adam/jpy_model_linaer/fifth_dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/fifth_dense/kernel/v* 
_output_shapes
:
*
dtype0
«
)Adam/jpy_model_linaer/fourth_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/jpy_model_linaer/fourth_dense/bias/v
€
=Adam/jpy_model_linaer/fourth_dense/bias/v/Read/ReadVariableOpReadVariableOp)Adam/jpy_model_linaer/fourth_dense/bias/v*
_output_shapes	
:*
dtype0
΄
+Adam/jpy_model_linaer/fourth_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+Adam/jpy_model_linaer/fourth_dense/kernel/v
­
?Adam/jpy_model_linaer/fourth_dense/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/jpy_model_linaer/fourth_dense/kernel/v* 
_output_shapes
:
*
dtype0
©
(Adam/jpy_model_linaer/third_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/third_dense/bias/v
’
<Adam/jpy_model_linaer/third_dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/third_dense/bias/v*
_output_shapes	
:*
dtype0
²
*Adam/jpy_model_linaer/third_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/jpy_model_linaer/third_dense/kernel/v
«
>Adam/jpy_model_linaer/third_dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/third_dense/kernel/v* 
_output_shapes
:
*
dtype0
«
)Adam/jpy_model_linaer/second_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/jpy_model_linaer/second_dense/bias/v
€
=Adam/jpy_model_linaer/second_dense/bias/v/Read/ReadVariableOpReadVariableOp)Adam/jpy_model_linaer/second_dense/bias/v*
_output_shapes	
:*
dtype0
΄
+Adam/jpy_model_linaer/second_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+Adam/jpy_model_linaer/second_dense/kernel/v
­
?Adam/jpy_model_linaer/second_dense/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/jpy_model_linaer/second_dense/kernel/v* 
_output_shapes
:
*
dtype0
©
(Adam/jpy_model_linaer/first_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/first_dense/bias/v
’
<Adam/jpy_model_linaer/first_dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/first_dense/bias/v*
_output_shapes	
:*
dtype0
±
*Adam/jpy_model_linaer/first_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/jpy_model_linaer/first_dense/kernel/v
ͺ
>Adam/jpy_model_linaer/first_dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/first_dense/kernel/v*
_output_shapes
:	*
dtype0
’
%Adam/jpy_model_linaer/output_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/jpy_model_linaer/output_1/bias/m

9Adam/jpy_model_linaer/output_1/bias/m/Read/ReadVariableOpReadVariableOp%Adam/jpy_model_linaer/output_1/bias/m*
_output_shapes
:*
dtype0
«
'Adam/jpy_model_linaer/output_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*8
shared_name)'Adam/jpy_model_linaer/output_1/kernel/m
€
;Adam/jpy_model_linaer/output_1/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/jpy_model_linaer/output_1/kernel/m*
_output_shapes
:	*
dtype0
©
(Adam/jpy_model_linaer/fifth_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/fifth_dense/bias/m
’
<Adam/jpy_model_linaer/fifth_dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/fifth_dense/bias/m*
_output_shapes	
:*
dtype0
²
*Adam/jpy_model_linaer/fifth_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/jpy_model_linaer/fifth_dense/kernel/m
«
>Adam/jpy_model_linaer/fifth_dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/fifth_dense/kernel/m* 
_output_shapes
:
*
dtype0
«
)Adam/jpy_model_linaer/fourth_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/jpy_model_linaer/fourth_dense/bias/m
€
=Adam/jpy_model_linaer/fourth_dense/bias/m/Read/ReadVariableOpReadVariableOp)Adam/jpy_model_linaer/fourth_dense/bias/m*
_output_shapes	
:*
dtype0
΄
+Adam/jpy_model_linaer/fourth_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+Adam/jpy_model_linaer/fourth_dense/kernel/m
­
?Adam/jpy_model_linaer/fourth_dense/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/jpy_model_linaer/fourth_dense/kernel/m* 
_output_shapes
:
*
dtype0
©
(Adam/jpy_model_linaer/third_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/third_dense/bias/m
’
<Adam/jpy_model_linaer/third_dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/third_dense/bias/m*
_output_shapes	
:*
dtype0
²
*Adam/jpy_model_linaer/third_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/jpy_model_linaer/third_dense/kernel/m
«
>Adam/jpy_model_linaer/third_dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/third_dense/kernel/m* 
_output_shapes
:
*
dtype0
«
)Adam/jpy_model_linaer/second_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/jpy_model_linaer/second_dense/bias/m
€
=Adam/jpy_model_linaer/second_dense/bias/m/Read/ReadVariableOpReadVariableOp)Adam/jpy_model_linaer/second_dense/bias/m*
_output_shapes	
:*
dtype0
΄
+Adam/jpy_model_linaer/second_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+Adam/jpy_model_linaer/second_dense/kernel/m
­
?Adam/jpy_model_linaer/second_dense/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/jpy_model_linaer/second_dense/kernel/m* 
_output_shapes
:
*
dtype0
©
(Adam/jpy_model_linaer/first_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/jpy_model_linaer/first_dense/bias/m
’
<Adam/jpy_model_linaer/first_dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/jpy_model_linaer/first_dense/bias/m*
_output_shapes	
:*
dtype0
±
*Adam/jpy_model_linaer/first_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/jpy_model_linaer/first_dense/kernel/m
ͺ
>Adam/jpy_model_linaer/first_dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/jpy_model_linaer/first_dense/kernel/m*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

jpy_model_linaer/output_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name jpy_model_linaer/output_1/bias

2jpy_model_linaer/output_1/bias/Read/ReadVariableOpReadVariableOpjpy_model_linaer/output_1/bias*
_output_shapes
:*
dtype0

 jpy_model_linaer/output_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" jpy_model_linaer/output_1/kernel

4jpy_model_linaer/output_1/kernel/Read/ReadVariableOpReadVariableOp jpy_model_linaer/output_1/kernel*
_output_shapes
:	*
dtype0

!jpy_model_linaer/fifth_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!jpy_model_linaer/fifth_dense/bias

5jpy_model_linaer/fifth_dense/bias/Read/ReadVariableOpReadVariableOp!jpy_model_linaer/fifth_dense/bias*
_output_shapes	
:*
dtype0
€
#jpy_model_linaer/fifth_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#jpy_model_linaer/fifth_dense/kernel

7jpy_model_linaer/fifth_dense/kernel/Read/ReadVariableOpReadVariableOp#jpy_model_linaer/fifth_dense/kernel* 
_output_shapes
:
*
dtype0

"jpy_model_linaer/fourth_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"jpy_model_linaer/fourth_dense/bias

6jpy_model_linaer/fourth_dense/bias/Read/ReadVariableOpReadVariableOp"jpy_model_linaer/fourth_dense/bias*
_output_shapes	
:*
dtype0
¦
$jpy_model_linaer/fourth_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$jpy_model_linaer/fourth_dense/kernel

8jpy_model_linaer/fourth_dense/kernel/Read/ReadVariableOpReadVariableOp$jpy_model_linaer/fourth_dense/kernel* 
_output_shapes
:
*
dtype0

!jpy_model_linaer/third_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!jpy_model_linaer/third_dense/bias

5jpy_model_linaer/third_dense/bias/Read/ReadVariableOpReadVariableOp!jpy_model_linaer/third_dense/bias*
_output_shapes	
:*
dtype0
€
#jpy_model_linaer/third_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#jpy_model_linaer/third_dense/kernel

7jpy_model_linaer/third_dense/kernel/Read/ReadVariableOpReadVariableOp#jpy_model_linaer/third_dense/kernel* 
_output_shapes
:
*
dtype0

"jpy_model_linaer/second_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"jpy_model_linaer/second_dense/bias

6jpy_model_linaer/second_dense/bias/Read/ReadVariableOpReadVariableOp"jpy_model_linaer/second_dense/bias*
_output_shapes	
:*
dtype0
¦
$jpy_model_linaer/second_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$jpy_model_linaer/second_dense/kernel

8jpy_model_linaer/second_dense/kernel/Read/ReadVariableOpReadVariableOp$jpy_model_linaer/second_dense/kernel* 
_output_shapes
:
*
dtype0

!jpy_model_linaer/first_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!jpy_model_linaer/first_dense/bias

5jpy_model_linaer/first_dense/bias/Read/ReadVariableOpReadVariableOp!jpy_model_linaer/first_dense/bias*
_output_shapes	
:*
dtype0
£
#jpy_model_linaer/first_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#jpy_model_linaer/first_dense/kernel

7jpy_model_linaer/first_dense/kernel/Read/ReadVariableOpReadVariableOp#jpy_model_linaer/first_dense/kernel*
_output_shapes
:	*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
ϊ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#jpy_model_linaer/first_dense/kernel!jpy_model_linaer/first_dense/bias$jpy_model_linaer/second_dense/kernel"jpy_model_linaer/second_dense/bias#jpy_model_linaer/third_dense/kernel!jpy_model_linaer/third_dense/bias$jpy_model_linaer/fourth_dense/kernel"jpy_model_linaer/fourth_dense/bias#jpy_model_linaer/fifth_dense/kernel!jpy_model_linaer/fifth_dense/bias jpy_model_linaer/output_1/kerneljpy_model_linaer/output_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_32171

NoOpNoOp
ήT
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*T
valueTBT BT
Ε
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
first_dense
	second_dense


first_drop
third_dense
fourth_dense
fifth_dense
first_output
	optimizer

signatures*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
°
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
"trace_0
#trace_1
$trace_2
%trace_3* 
6
&trace_0
'trace_1
(trace_2
)trace_3* 
* 
¦
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*
¦
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
₯
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator* 
¦
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias*
¦
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias*
¦
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
bias*
¦
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

kernel
bias*
΄
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratemmmmmmmm m‘m’m£m€v₯v¦v§v¨v©vͺv«v¬v­v?v―v°*

Zserving_default* 
c]
VARIABLE_VALUE#jpy_model_linaer/first_dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!jpy_model_linaer/first_dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$jpy_model_linaer/second_dense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"jpy_model_linaer/second_dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#jpy_model_linaer/third_dense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!jpy_model_linaer/third_dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$jpy_model_linaer/fourth_dense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"jpy_model_linaer/fourth_dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#jpy_model_linaer/fifth_dense/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!jpy_model_linaer/fifth_dense/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE jpy_model_linaer/output_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEjpy_model_linaer/output_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
	1

2
3
4
5
6*

[0
\1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 

0
1*

0
1*
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
* 
* 
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

ptrace_0
qtrace_1* 

rtrace_0
strace_1* 
* 

0
1*

0
1*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 

0
1*

0
1*
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

trace_0* 

trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUE*Adam/jpy_model_linaer/first_dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/first_dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/jpy_model_linaer/second_dense/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/jpy_model_linaer/second_dense/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/jpy_model_linaer/third_dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/third_dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/jpy_model_linaer/fourth_dense/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/jpy_model_linaer/fourth_dense/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/jpy_model_linaer/fifth_dense/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/fifth_dense/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/jpy_model_linaer/output_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/jpy_model_linaer/output_1/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/jpy_model_linaer/first_dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/first_dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/jpy_model_linaer/second_dense/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/jpy_model_linaer/second_dense/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/jpy_model_linaer/third_dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/third_dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/jpy_model_linaer/fourth_dense/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/jpy_model_linaer/fourth_dense/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/jpy_model_linaer/fifth_dense/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/jpy_model_linaer/fifth_dense/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/jpy_model_linaer/output_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/jpy_model_linaer/output_1/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ε
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7jpy_model_linaer/first_dense/kernel/Read/ReadVariableOp5jpy_model_linaer/first_dense/bias/Read/ReadVariableOp8jpy_model_linaer/second_dense/kernel/Read/ReadVariableOp6jpy_model_linaer/second_dense/bias/Read/ReadVariableOp7jpy_model_linaer/third_dense/kernel/Read/ReadVariableOp5jpy_model_linaer/third_dense/bias/Read/ReadVariableOp8jpy_model_linaer/fourth_dense/kernel/Read/ReadVariableOp6jpy_model_linaer/fourth_dense/bias/Read/ReadVariableOp7jpy_model_linaer/fifth_dense/kernel/Read/ReadVariableOp5jpy_model_linaer/fifth_dense/bias/Read/ReadVariableOp4jpy_model_linaer/output_1/kernel/Read/ReadVariableOp2jpy_model_linaer/output_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp>Adam/jpy_model_linaer/first_dense/kernel/m/Read/ReadVariableOp<Adam/jpy_model_linaer/first_dense/bias/m/Read/ReadVariableOp?Adam/jpy_model_linaer/second_dense/kernel/m/Read/ReadVariableOp=Adam/jpy_model_linaer/second_dense/bias/m/Read/ReadVariableOp>Adam/jpy_model_linaer/third_dense/kernel/m/Read/ReadVariableOp<Adam/jpy_model_linaer/third_dense/bias/m/Read/ReadVariableOp?Adam/jpy_model_linaer/fourth_dense/kernel/m/Read/ReadVariableOp=Adam/jpy_model_linaer/fourth_dense/bias/m/Read/ReadVariableOp>Adam/jpy_model_linaer/fifth_dense/kernel/m/Read/ReadVariableOp<Adam/jpy_model_linaer/fifth_dense/bias/m/Read/ReadVariableOp;Adam/jpy_model_linaer/output_1/kernel/m/Read/ReadVariableOp9Adam/jpy_model_linaer/output_1/bias/m/Read/ReadVariableOp>Adam/jpy_model_linaer/first_dense/kernel/v/Read/ReadVariableOp<Adam/jpy_model_linaer/first_dense/bias/v/Read/ReadVariableOp?Adam/jpy_model_linaer/second_dense/kernel/v/Read/ReadVariableOp=Adam/jpy_model_linaer/second_dense/bias/v/Read/ReadVariableOp>Adam/jpy_model_linaer/third_dense/kernel/v/Read/ReadVariableOp<Adam/jpy_model_linaer/third_dense/bias/v/Read/ReadVariableOp?Adam/jpy_model_linaer/fourth_dense/kernel/v/Read/ReadVariableOp=Adam/jpy_model_linaer/fourth_dense/bias/v/Read/ReadVariableOp>Adam/jpy_model_linaer/fifth_dense/kernel/v/Read/ReadVariableOp<Adam/jpy_model_linaer/fifth_dense/bias/v/Read/ReadVariableOp;Adam/jpy_model_linaer/output_1/kernel/v/Read/ReadVariableOp9Adam/jpy_model_linaer/output_1/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_32632
ά
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#jpy_model_linaer/first_dense/kernel!jpy_model_linaer/first_dense/bias$jpy_model_linaer/second_dense/kernel"jpy_model_linaer/second_dense/bias#jpy_model_linaer/third_dense/kernel!jpy_model_linaer/third_dense/bias$jpy_model_linaer/fourth_dense/kernel"jpy_model_linaer/fourth_dense/bias#jpy_model_linaer/fifth_dense/kernel!jpy_model_linaer/fifth_dense/bias jpy_model_linaer/output_1/kerneljpy_model_linaer/output_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount*Adam/jpy_model_linaer/first_dense/kernel/m(Adam/jpy_model_linaer/first_dense/bias/m+Adam/jpy_model_linaer/second_dense/kernel/m)Adam/jpy_model_linaer/second_dense/bias/m*Adam/jpy_model_linaer/third_dense/kernel/m(Adam/jpy_model_linaer/third_dense/bias/m+Adam/jpy_model_linaer/fourth_dense/kernel/m)Adam/jpy_model_linaer/fourth_dense/bias/m*Adam/jpy_model_linaer/fifth_dense/kernel/m(Adam/jpy_model_linaer/fifth_dense/bias/m'Adam/jpy_model_linaer/output_1/kernel/m%Adam/jpy_model_linaer/output_1/bias/m*Adam/jpy_model_linaer/first_dense/kernel/v(Adam/jpy_model_linaer/first_dense/bias/v+Adam/jpy_model_linaer/second_dense/kernel/v)Adam/jpy_model_linaer/second_dense/bias/v*Adam/jpy_model_linaer/third_dense/kernel/v(Adam/jpy_model_linaer/third_dense/bias/v+Adam/jpy_model_linaer/fourth_dense/kernel/v)Adam/jpy_model_linaer/fourth_dense/bias/v*Adam/jpy_model_linaer/fifth_dense/kernel/v(Adam/jpy_model_linaer/fifth_dense/bias/v'Adam/jpy_model_linaer/output_1/kernel/v%Adam/jpy_model_linaer/output_1/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_32777ύ

Ό
0__inference_jpy_model_linaer_layer_call_fn_32064
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
₯

ω
F__inference_first_dense_layer_call_and_return_conditional_losses_32348

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

,__inference_second_dense_layer_call_fn_32357

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_second_dense_layer_call_and_return_conditional_losses_31751p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ώ
f
-__inference_first_dropout_layer_call_fn_32378

inputs
identity’StatefulPartitionedCallΗ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31919p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
©

ϊ
F__inference_third_dense_layer_call_and_return_conditional_losses_32415

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Έ8


K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32275

inputs=
*first_dense_matmul_readvariableop_resource:	:
+first_dense_biasadd_readvariableop_resource:	?
+second_dense_matmul_readvariableop_resource:
;
,second_dense_biasadd_readvariableop_resource:	>
*third_dense_matmul_readvariableop_resource:
:
+third_dense_biasadd_readvariableop_resource:	?
+fourth_dense_matmul_readvariableop_resource:
;
,fourth_dense_biasadd_readvariableop_resource:	>
*fifth_dense_matmul_readvariableop_resource:
:
+fifth_dense_biasadd_readvariableop_resource:	:
'output_1_matmul_readvariableop_resource:	6
(output_1_biasadd_readvariableop_resource:
identity’"fifth_dense/BiasAdd/ReadVariableOp’!fifth_dense/MatMul/ReadVariableOp’"first_dense/BiasAdd/ReadVariableOp’!first_dense/MatMul/ReadVariableOp’#fourth_dense/BiasAdd/ReadVariableOp’"fourth_dense/MatMul/ReadVariableOp’output_1/BiasAdd/ReadVariableOp’output_1/MatMul/ReadVariableOp’#second_dense/BiasAdd/ReadVariableOp’"second_dense/MatMul/ReadVariableOp’"third_dense/BiasAdd/ReadVariableOp’!third_dense/MatMul/ReadVariableOp
!first_dense/MatMul/ReadVariableOpReadVariableOp*first_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
first_dense/MatMulMatMulinputs)first_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"first_dense/BiasAdd/ReadVariableOpReadVariableOp+first_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
first_dense/BiasAddBiasAddfirst_dense/MatMul:product:0*first_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
first_dense/ReluRelufirst_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
"second_dense/MatMul/ReadVariableOpReadVariableOp+second_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
second_dense/MatMulMatMulfirst_dense/Relu:activations:0*second_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#second_dense/BiasAdd/ReadVariableOpReadVariableOp,second_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
second_dense/BiasAddBiasAddsecond_dense/MatMul:product:0+second_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????k
second_dense/ReluRelusecond_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????v
first_dropout/IdentityIdentitysecond_dense/Relu:activations:0*
T0*(
_output_shapes
:?????????
!third_dense/MatMul/ReadVariableOpReadVariableOp*third_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
third_dense/MatMulMatMulfirst_dropout/Identity:output:0)third_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"third_dense/BiasAdd/ReadVariableOpReadVariableOp+third_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
third_dense/BiasAddBiasAddthird_dense/MatMul:product:0*third_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
third_dense/ReluReluthird_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
"fourth_dense/MatMul/ReadVariableOpReadVariableOp+fourth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
fourth_dense/MatMulMatMulthird_dense/Relu:activations:0*fourth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#fourth_dense/BiasAdd/ReadVariableOpReadVariableOp,fourth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fourth_dense/BiasAddBiasAddfourth_dense/MatMul:product:0+fourth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????k
fourth_dense/ReluRelufourth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!fifth_dense/MatMul/ReadVariableOpReadVariableOp*fifth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
fifth_dense/MatMulMatMulfourth_dense/Relu:activations:0)fifth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"fifth_dense/BiasAdd/ReadVariableOpReadVariableOp+fifth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fifth_dense/BiasAddBiasAddfifth_dense/MatMul:product:0*fifth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
fifth_dense/ReluRelufifth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
output_1/MatMul/ReadVariableOpReadVariableOp'output_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
output_1/MatMulMatMulfifth_dense/Relu:activations:0&output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
output_1/BiasAdd/ReadVariableOpReadVariableOp(output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_1/BiasAddBiasAddoutput_1/MatMul:product:0'output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentityoutput_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ϊ
NoOpNoOp#^fifth_dense/BiasAdd/ReadVariableOp"^fifth_dense/MatMul/ReadVariableOp#^first_dense/BiasAdd/ReadVariableOp"^first_dense/MatMul/ReadVariableOp$^fourth_dense/BiasAdd/ReadVariableOp#^fourth_dense/MatMul/ReadVariableOp ^output_1/BiasAdd/ReadVariableOp^output_1/MatMul/ReadVariableOp$^second_dense/BiasAdd/ReadVariableOp#^second_dense/MatMul/ReadVariableOp#^third_dense/BiasAdd/ReadVariableOp"^third_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2H
"fifth_dense/BiasAdd/ReadVariableOp"fifth_dense/BiasAdd/ReadVariableOp2F
!fifth_dense/MatMul/ReadVariableOp!fifth_dense/MatMul/ReadVariableOp2H
"first_dense/BiasAdd/ReadVariableOp"first_dense/BiasAdd/ReadVariableOp2F
!first_dense/MatMul/ReadVariableOp!first_dense/MatMul/ReadVariableOp2J
#fourth_dense/BiasAdd/ReadVariableOp#fourth_dense/BiasAdd/ReadVariableOp2H
"fourth_dense/MatMul/ReadVariableOp"fourth_dense/MatMul/ReadVariableOp2B
output_1/BiasAdd/ReadVariableOpoutput_1/BiasAdd/ReadVariableOp2@
output_1/MatMul/ReadVariableOpoutput_1/MatMul/ReadVariableOp2J
#second_dense/BiasAdd/ReadVariableOp#second_dense/BiasAdd/ReadVariableOp2H
"second_dense/MatMul/ReadVariableOp"second_dense/MatMul/ReadVariableOp2H
"third_dense/BiasAdd/ReadVariableOp"third_dense/BiasAdd/ReadVariableOp2F
!third_dense/MatMul/ReadVariableOp!third_dense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ν$
?
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32099
input_1$
first_dense_32067:	 
first_dense_32069:	&
second_dense_32072:
!
second_dense_32074:	%
third_dense_32078:
 
third_dense_32080:	&
fourth_dense_32083:
!
fourth_dense_32085:	%
fifth_dense_32088:
 
fifth_dense_32090:	!
output_1_32093:	
output_1_32095:
identity’#fifth_dense/StatefulPartitionedCall’#first_dense/StatefulPartitionedCall’$fourth_dense/StatefulPartitionedCall’ output_1/StatefulPartitionedCall’$second_dense/StatefulPartitionedCall’#third_dense/StatefulPartitionedCallώ
#first_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1first_dense_32067first_dense_32069*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_first_dense_layer_call_and_return_conditional_losses_31734§
$second_dense/StatefulPartitionedCallStatefulPartitionedCall,first_dense/StatefulPartitionedCall:output:0second_dense_32072second_dense_32074*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_second_dense_layer_call_and_return_conditional_losses_31751μ
first_dropout/PartitionedCallPartitionedCall-second_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31762
#third_dense/StatefulPartitionedCallStatefulPartitionedCall&first_dropout/PartitionedCall:output:0third_dense_32078third_dense_32080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_third_dense_layer_call_and_return_conditional_losses_31775§
$fourth_dense/StatefulPartitionedCallStatefulPartitionedCall,third_dense/StatefulPartitionedCall:output:0fourth_dense_32083fourth_dense_32085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792€
#fifth_dense/StatefulPartitionedCallStatefulPartitionedCall-fourth_dense/StatefulPartitionedCall:output:0fifth_dense_32088fifth_dense_32090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809
 output_1/StatefulPartitionedCallStatefulPartitionedCall,fifth_dense/StatefulPartitionedCall:output:0output_1_32093output_1_32095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_1_layer_call_and_return_conditional_losses_31825x
IdentityIdentity)output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????©
NoOpNoOp$^fifth_dense/StatefulPartitionedCall$^first_dense/StatefulPartitionedCall%^fourth_dense/StatefulPartitionedCall!^output_1/StatefulPartitionedCall%^second_dense/StatefulPartitionedCall$^third_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2J
#fifth_dense/StatefulPartitionedCall#fifth_dense/StatefulPartitionedCall2J
#first_dense/StatefulPartitionedCall#first_dense/StatefulPartitionedCall2L
$fourth_dense/StatefulPartitionedCall$fourth_dense/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2L
$second_dense/StatefulPartitionedCall$second_dense/StatefulPartitionedCall2J
#third_dense/StatefulPartitionedCall#third_dense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
©

ϊ
F__inference_third_dense_layer_call_and_return_conditional_losses_31775

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
κ$
ώ
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_31832

inputs$
first_dense_31735:	 
first_dense_31737:	&
second_dense_31752:
!
second_dense_31754:	%
third_dense_31776:
 
third_dense_31778:	&
fourth_dense_31793:
!
fourth_dense_31795:	%
fifth_dense_31810:
 
fifth_dense_31812:	!
output_1_31826:	
output_1_31828:
identity’#fifth_dense/StatefulPartitionedCall’#first_dense/StatefulPartitionedCall’$fourth_dense/StatefulPartitionedCall’ output_1/StatefulPartitionedCall’$second_dense/StatefulPartitionedCall’#third_dense/StatefulPartitionedCallύ
#first_dense/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_dense_31735first_dense_31737*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_first_dense_layer_call_and_return_conditional_losses_31734§
$second_dense/StatefulPartitionedCallStatefulPartitionedCall,first_dense/StatefulPartitionedCall:output:0second_dense_31752second_dense_31754*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_second_dense_layer_call_and_return_conditional_losses_31751μ
first_dropout/PartitionedCallPartitionedCall-second_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31762
#third_dense/StatefulPartitionedCallStatefulPartitionedCall&first_dropout/PartitionedCall:output:0third_dense_31776third_dense_31778*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_third_dense_layer_call_and_return_conditional_losses_31775§
$fourth_dense/StatefulPartitionedCallStatefulPartitionedCall,third_dense/StatefulPartitionedCall:output:0fourth_dense_31793fourth_dense_31795*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792€
#fifth_dense/StatefulPartitionedCallStatefulPartitionedCall-fourth_dense/StatefulPartitionedCall:output:0fifth_dense_31810fifth_dense_31812*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809
 output_1/StatefulPartitionedCallStatefulPartitionedCall,fifth_dense/StatefulPartitionedCall:output:0output_1_31826output_1_31828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_1_layer_call_and_return_conditional_losses_31825x
IdentityIdentity)output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????©
NoOpNoOp$^fifth_dense/StatefulPartitionedCall$^first_dense/StatefulPartitionedCall%^fourth_dense/StatefulPartitionedCall!^output_1/StatefulPartitionedCall%^second_dense/StatefulPartitionedCall$^third_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2J
#fifth_dense/StatefulPartitionedCall#fifth_dense/StatefulPartitionedCall2J
#first_dense/StatefulPartitionedCall#first_dense/StatefulPartitionedCall2L
$fourth_dense/StatefulPartitionedCall$fourth_dense/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2L
$second_dense/StatefulPartitionedCall$second_dense/StatefulPartitionedCall2J
#third_dense/StatefulPartitionedCall#third_dense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ	
υ
C__inference_output_1_layer_call_and_return_conditional_losses_31825

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
―Ί
π 
!__inference__traced_restore_32777
file_prefixG
4assignvariableop_jpy_model_linaer_first_dense_kernel:	C
4assignvariableop_1_jpy_model_linaer_first_dense_bias:	K
7assignvariableop_2_jpy_model_linaer_second_dense_kernel:
D
5assignvariableop_3_jpy_model_linaer_second_dense_bias:	J
6assignvariableop_4_jpy_model_linaer_third_dense_kernel:
C
4assignvariableop_5_jpy_model_linaer_third_dense_bias:	K
7assignvariableop_6_jpy_model_linaer_fourth_dense_kernel:
D
5assignvariableop_7_jpy_model_linaer_fourth_dense_bias:	J
6assignvariableop_8_jpy_model_linaer_fifth_dense_kernel:
C
4assignvariableop_9_jpy_model_linaer_fifth_dense_bias:	G
4assignvariableop_10_jpy_model_linaer_output_1_kernel:	@
2assignvariableop_11_jpy_model_linaer_output_1_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: Q
>assignvariableop_21_adam_jpy_model_linaer_first_dense_kernel_m:	K
<assignvariableop_22_adam_jpy_model_linaer_first_dense_bias_m:	S
?assignvariableop_23_adam_jpy_model_linaer_second_dense_kernel_m:
L
=assignvariableop_24_adam_jpy_model_linaer_second_dense_bias_m:	R
>assignvariableop_25_adam_jpy_model_linaer_third_dense_kernel_m:
K
<assignvariableop_26_adam_jpy_model_linaer_third_dense_bias_m:	S
?assignvariableop_27_adam_jpy_model_linaer_fourth_dense_kernel_m:
L
=assignvariableop_28_adam_jpy_model_linaer_fourth_dense_bias_m:	R
>assignvariableop_29_adam_jpy_model_linaer_fifth_dense_kernel_m:
K
<assignvariableop_30_adam_jpy_model_linaer_fifth_dense_bias_m:	N
;assignvariableop_31_adam_jpy_model_linaer_output_1_kernel_m:	G
9assignvariableop_32_adam_jpy_model_linaer_output_1_bias_m:Q
>assignvariableop_33_adam_jpy_model_linaer_first_dense_kernel_v:	K
<assignvariableop_34_adam_jpy_model_linaer_first_dense_bias_v:	S
?assignvariableop_35_adam_jpy_model_linaer_second_dense_kernel_v:
L
=assignvariableop_36_adam_jpy_model_linaer_second_dense_bias_v:	R
>assignvariableop_37_adam_jpy_model_linaer_third_dense_kernel_v:
K
<assignvariableop_38_adam_jpy_model_linaer_third_dense_bias_v:	S
?assignvariableop_39_adam_jpy_model_linaer_fourth_dense_kernel_v:
L
=assignvariableop_40_adam_jpy_model_linaer_fourth_dense_bias_v:	R
>assignvariableop_41_adam_jpy_model_linaer_fifth_dense_kernel_v:
K
<assignvariableop_42_adam_jpy_model_linaer_fifth_dense_bias_v:	N
;assignvariableop_43_adam_jpy_model_linaer_output_1_kernel_v:	G
9assignvariableop_44_adam_jpy_model_linaer_output_1_bias_v:
identity_46’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ά
value¬B©.B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΜ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ξ
_output_shapes»
Έ::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp4assignvariableop_jpy_model_linaer_first_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_1AssignVariableOp4assignvariableop_1_jpy_model_linaer_first_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_2AssignVariableOp7assignvariableop_2_jpy_model_linaer_second_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_3AssignVariableOp5assignvariableop_3_jpy_model_linaer_second_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_4AssignVariableOp6assignvariableop_4_jpy_model_linaer_third_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_5AssignVariableOp4assignvariableop_5_jpy_model_linaer_third_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_6AssignVariableOp7assignvariableop_6_jpy_model_linaer_fourth_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_7AssignVariableOp5assignvariableop_7_jpy_model_linaer_fourth_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_8AssignVariableOp6assignvariableop_8_jpy_model_linaer_fifth_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_9AssignVariableOp4assignvariableop_9_jpy_model_linaer_fifth_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_10AssignVariableOp4assignvariableop_10_jpy_model_linaer_output_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_11AssignVariableOp2assignvariableop_11_jpy_model_linaer_output_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_jpy_model_linaer_first_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_22AssignVariableOp<assignvariableop_22_adam_jpy_model_linaer_first_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_23AssignVariableOp?assignvariableop_23_adam_jpy_model_linaer_second_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp=assignvariableop_24_adam_jpy_model_linaer_second_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_25AssignVariableOp>assignvariableop_25_adam_jpy_model_linaer_third_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_26AssignVariableOp<assignvariableop_26_adam_jpy_model_linaer_third_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_27AssignVariableOp?assignvariableop_27_adam_jpy_model_linaer_fourth_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp=assignvariableop_28_adam_jpy_model_linaer_fourth_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_jpy_model_linaer_fifth_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_30AssignVariableOp<assignvariableop_30_adam_jpy_model_linaer_fifth_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_31AssignVariableOp;assignvariableop_31_adam_jpy_model_linaer_output_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_32AssignVariableOp9assignvariableop_32_adam_jpy_model_linaer_output_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_jpy_model_linaer_first_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_34AssignVariableOp<assignvariableop_34_adam_jpy_model_linaer_first_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_35AssignVariableOp?assignvariableop_35_adam_jpy_model_linaer_second_dense_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp=assignvariableop_36_adam_jpy_model_linaer_second_dense_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_jpy_model_linaer_third_dense_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_38AssignVariableOp<assignvariableop_38_adam_jpy_model_linaer_third_dense_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_39AssignVariableOp?assignvariableop_39_adam_jpy_model_linaer_fourth_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_jpy_model_linaer_fourth_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_41AssignVariableOp>assignvariableop_41_adam_jpy_model_linaer_fifth_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_42AssignVariableOp<assignvariableop_42_adam_jpy_model_linaer_fifth_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_43AssignVariableOp;assignvariableop_43_adam_jpy_model_linaer_output_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_44AssignVariableOp9assignvariableop_44_adam_jpy_model_linaer_output_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Κ	
υ
C__inference_output_1_layer_call_and_return_conditional_losses_32474

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

Ό
0__inference_jpy_model_linaer_layer_call_fn_31859
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_31832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
¬
I
-__inference_first_dropout_layer_call_fn_32373

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31762a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯

ω
F__inference_first_dense_layer_call_and_return_conditional_losses_31734

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ν

+__inference_first_dense_layer_call_fn_32337

inputs
unknown:	
	unknown_0:	
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_first_dense_layer_call_and_return_conditional_losses_31734p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ

ϋ
G__inference_second_dense_layer_call_and_return_conditional_losses_32368

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
©

ϊ
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
©

ϊ
F__inference_fifth_dense_layer_call_and_return_conditional_losses_32455

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
’&
¦
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32008

inputs$
first_dense_31976:	 
first_dense_31978:	&
second_dense_31981:
!
second_dense_31983:	%
third_dense_31987:
 
third_dense_31989:	&
fourth_dense_31992:
!
fourth_dense_31994:	%
fifth_dense_31997:
 
fifth_dense_31999:	!
output_1_32002:	
output_1_32004:
identity’#fifth_dense/StatefulPartitionedCall’#first_dense/StatefulPartitionedCall’%first_dropout/StatefulPartitionedCall’$fourth_dense/StatefulPartitionedCall’ output_1/StatefulPartitionedCall’$second_dense/StatefulPartitionedCall’#third_dense/StatefulPartitionedCallύ
#first_dense/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_dense_31976first_dense_31978*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_first_dense_layer_call_and_return_conditional_losses_31734§
$second_dense/StatefulPartitionedCallStatefulPartitionedCall,first_dense/StatefulPartitionedCall:output:0second_dense_31981second_dense_31983*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_second_dense_layer_call_and_return_conditional_losses_31751ό
%first_dropout/StatefulPartitionedCallStatefulPartitionedCall-second_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31919₯
#third_dense/StatefulPartitionedCallStatefulPartitionedCall.first_dropout/StatefulPartitionedCall:output:0third_dense_31987third_dense_31989*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_third_dense_layer_call_and_return_conditional_losses_31775§
$fourth_dense/StatefulPartitionedCallStatefulPartitionedCall,third_dense/StatefulPartitionedCall:output:0fourth_dense_31992fourth_dense_31994*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792€
#fifth_dense/StatefulPartitionedCallStatefulPartitionedCall-fourth_dense/StatefulPartitionedCall:output:0fifth_dense_31997fifth_dense_31999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809
 output_1/StatefulPartitionedCallStatefulPartitionedCall,fifth_dense/StatefulPartitionedCall:output:0output_1_32002output_1_32004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_1_layer_call_and_return_conditional_losses_31825x
IdentityIdentity)output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ρ
NoOpNoOp$^fifth_dense/StatefulPartitionedCall$^first_dense/StatefulPartitionedCall&^first_dropout/StatefulPartitionedCall%^fourth_dense/StatefulPartitionedCall!^output_1/StatefulPartitionedCall%^second_dense/StatefulPartitionedCall$^third_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2J
#fifth_dense/StatefulPartitionedCall#fifth_dense/StatefulPartitionedCall2J
#first_dense/StatefulPartitionedCall#first_dense/StatefulPartitionedCall2N
%first_dropout/StatefulPartitionedCall%first_dropout/StatefulPartitionedCall2L
$fourth_dense/StatefulPartitionedCall$fourth_dense/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2L
$second_dense/StatefulPartitionedCall$second_dense/StatefulPartitionedCall2J
#third_dense/StatefulPartitionedCall#third_dense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ό@


K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32328

inputs=
*first_dense_matmul_readvariableop_resource:	:
+first_dense_biasadd_readvariableop_resource:	?
+second_dense_matmul_readvariableop_resource:
;
,second_dense_biasadd_readvariableop_resource:	>
*third_dense_matmul_readvariableop_resource:
:
+third_dense_biasadd_readvariableop_resource:	?
+fourth_dense_matmul_readvariableop_resource:
;
,fourth_dense_biasadd_readvariableop_resource:	>
*fifth_dense_matmul_readvariableop_resource:
:
+fifth_dense_biasadd_readvariableop_resource:	:
'output_1_matmul_readvariableop_resource:	6
(output_1_biasadd_readvariableop_resource:
identity’"fifth_dense/BiasAdd/ReadVariableOp’!fifth_dense/MatMul/ReadVariableOp’"first_dense/BiasAdd/ReadVariableOp’!first_dense/MatMul/ReadVariableOp’#fourth_dense/BiasAdd/ReadVariableOp’"fourth_dense/MatMul/ReadVariableOp’output_1/BiasAdd/ReadVariableOp’output_1/MatMul/ReadVariableOp’#second_dense/BiasAdd/ReadVariableOp’"second_dense/MatMul/ReadVariableOp’"third_dense/BiasAdd/ReadVariableOp’!third_dense/MatMul/ReadVariableOp
!first_dense/MatMul/ReadVariableOpReadVariableOp*first_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
first_dense/MatMulMatMulinputs)first_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"first_dense/BiasAdd/ReadVariableOpReadVariableOp+first_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
first_dense/BiasAddBiasAddfirst_dense/MatMul:product:0*first_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
first_dense/ReluRelufirst_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
"second_dense/MatMul/ReadVariableOpReadVariableOp+second_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
second_dense/MatMulMatMulfirst_dense/Relu:activations:0*second_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#second_dense/BiasAdd/ReadVariableOpReadVariableOp,second_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
second_dense/BiasAddBiasAddsecond_dense/MatMul:product:0+second_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????k
second_dense/ReluRelusecond_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????`
first_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?
first_dropout/dropout/MulMulsecond_dense/Relu:activations:0$first_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????j
first_dropout/dropout/ShapeShapesecond_dense/Relu:activations:0*
T0*
_output_shapes
:©
2first_dropout/dropout/random_uniform/RandomUniformRandomUniform$first_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0i
$first_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Ρ
"first_dropout/dropout/GreaterEqualGreaterEqual;first_dropout/dropout/random_uniform/RandomUniform:output:0-first_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????
first_dropout/dropout/CastCast&first_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????
first_dropout/dropout/Mul_1Mulfirst_dropout/dropout/Mul:z:0first_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????
!third_dense/MatMul/ReadVariableOpReadVariableOp*third_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
third_dense/MatMulMatMulfirst_dropout/dropout/Mul_1:z:0)third_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"third_dense/BiasAdd/ReadVariableOpReadVariableOp+third_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
third_dense/BiasAddBiasAddthird_dense/MatMul:product:0*third_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
third_dense/ReluReluthird_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
"fourth_dense/MatMul/ReadVariableOpReadVariableOp+fourth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
fourth_dense/MatMulMatMulthird_dense/Relu:activations:0*fourth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#fourth_dense/BiasAdd/ReadVariableOpReadVariableOp,fourth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fourth_dense/BiasAddBiasAddfourth_dense/MatMul:product:0+fourth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????k
fourth_dense/ReluRelufourth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!fifth_dense/MatMul/ReadVariableOpReadVariableOp*fifth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
fifth_dense/MatMulMatMulfourth_dense/Relu:activations:0)fifth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"fifth_dense/BiasAdd/ReadVariableOpReadVariableOp+fifth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fifth_dense/BiasAddBiasAddfifth_dense/MatMul:product:0*fifth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
fifth_dense/ReluRelufifth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
output_1/MatMul/ReadVariableOpReadVariableOp'output_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
output_1/MatMulMatMulfifth_dense/Relu:activations:0&output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
output_1/BiasAdd/ReadVariableOpReadVariableOp(output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_1/BiasAddBiasAddoutput_1/MatMul:product:0'output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentityoutput_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ϊ
NoOpNoOp#^fifth_dense/BiasAdd/ReadVariableOp"^fifth_dense/MatMul/ReadVariableOp#^first_dense/BiasAdd/ReadVariableOp"^first_dense/MatMul/ReadVariableOp$^fourth_dense/BiasAdd/ReadVariableOp#^fourth_dense/MatMul/ReadVariableOp ^output_1/BiasAdd/ReadVariableOp^output_1/MatMul/ReadVariableOp$^second_dense/BiasAdd/ReadVariableOp#^second_dense/MatMul/ReadVariableOp#^third_dense/BiasAdd/ReadVariableOp"^third_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2H
"fifth_dense/BiasAdd/ReadVariableOp"fifth_dense/BiasAdd/ReadVariableOp2F
!fifth_dense/MatMul/ReadVariableOp!fifth_dense/MatMul/ReadVariableOp2H
"first_dense/BiasAdd/ReadVariableOp"first_dense/BiasAdd/ReadVariableOp2F
!first_dense/MatMul/ReadVariableOp!first_dense/MatMul/ReadVariableOp2J
#fourth_dense/BiasAdd/ReadVariableOp#fourth_dense/BiasAdd/ReadVariableOp2H
"fourth_dense/MatMul/ReadVariableOp"fourth_dense/MatMul/ReadVariableOp2B
output_1/BiasAdd/ReadVariableOpoutput_1/BiasAdd/ReadVariableOp2@
output_1/MatMul/ReadVariableOpoutput_1/MatMul/ReadVariableOp2J
#second_dense/BiasAdd/ReadVariableOp#second_dense/BiasAdd/ReadVariableOp2H
"second_dense/MatMul/ReadVariableOp"second_dense/MatMul/ReadVariableOp2H
"third_dense/BiasAdd/ReadVariableOp"third_dense/BiasAdd/ReadVariableOp2F
!third_dense/MatMul/ReadVariableOp!third_dense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ί
f
H__inference_first_dropout_layer_call_and_return_conditional_losses_32383

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
?

,__inference_fourth_dense_layer_call_fn_32424

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ

ϋ
G__inference_second_dense_layer_call_and_return_conditional_losses_31751

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ί
f
H__inference_first_dropout_layer_call_and_return_conditional_losses_31762

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Π

+__inference_third_dense_layer_call_fn_32404

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_third_dense_layer_call_and_return_conditional_losses_31775p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ώ	
g
H__inference_first_dropout_layer_call_and_return_conditional_losses_31919

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

»
0__inference_jpy_model_linaer_layer_call_fn_32229

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity’StatefulPartitionedCallζ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ

ϋ
G__inference_fourth_dense_layer_call_and_return_conditional_losses_32435

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
₯&
§
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32134
input_1$
first_dense_32102:	 
first_dense_32104:	&
second_dense_32107:
!
second_dense_32109:	%
third_dense_32113:
 
third_dense_32115:	&
fourth_dense_32118:
!
fourth_dense_32120:	%
fifth_dense_32123:
 
fifth_dense_32125:	!
output_1_32128:	
output_1_32130:
identity’#fifth_dense/StatefulPartitionedCall’#first_dense/StatefulPartitionedCall’%first_dropout/StatefulPartitionedCall’$fourth_dense/StatefulPartitionedCall’ output_1/StatefulPartitionedCall’$second_dense/StatefulPartitionedCall’#third_dense/StatefulPartitionedCallώ
#first_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1first_dense_32102first_dense_32104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_first_dense_layer_call_and_return_conditional_losses_31734§
$second_dense/StatefulPartitionedCallStatefulPartitionedCall,first_dense/StatefulPartitionedCall:output:0second_dense_32107second_dense_32109*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_second_dense_layer_call_and_return_conditional_losses_31751ό
%first_dropout/StatefulPartitionedCallStatefulPartitionedCall-second_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_first_dropout_layer_call_and_return_conditional_losses_31919₯
#third_dense/StatefulPartitionedCallStatefulPartitionedCall.first_dropout/StatefulPartitionedCall:output:0third_dense_32113third_dense_32115*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_third_dense_layer_call_and_return_conditional_losses_31775§
$fourth_dense/StatefulPartitionedCallStatefulPartitionedCall,third_dense/StatefulPartitionedCall:output:0fourth_dense_32118fourth_dense_32120*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792€
#fifth_dense/StatefulPartitionedCallStatefulPartitionedCall-fourth_dense/StatefulPartitionedCall:output:0fifth_dense_32123fifth_dense_32125*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809
 output_1/StatefulPartitionedCallStatefulPartitionedCall,fifth_dense/StatefulPartitionedCall:output:0output_1_32128output_1_32130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_1_layer_call_and_return_conditional_losses_31825x
IdentityIdentity)output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ρ
NoOpNoOp$^fifth_dense/StatefulPartitionedCall$^first_dense/StatefulPartitionedCall&^first_dropout/StatefulPartitionedCall%^fourth_dense/StatefulPartitionedCall!^output_1/StatefulPartitionedCall%^second_dense/StatefulPartitionedCall$^third_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2J
#fifth_dense/StatefulPartitionedCall#fifth_dense/StatefulPartitionedCall2J
#first_dense/StatefulPartitionedCall#first_dense/StatefulPartitionedCall2N
%first_dropout/StatefulPartitionedCall%first_dropout/StatefulPartitionedCall2L
$fourth_dense/StatefulPartitionedCall$fourth_dense/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2L
$second_dense/StatefulPartitionedCall$second_dense/StatefulPartitionedCall2J
#third_dense/StatefulPartitionedCall#third_dense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
Π

+__inference_fifth_dense_layer_call_fn_32444

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_fifth_dense_layer_call_and_return_conditional_losses_31809p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

»
0__inference_jpy_model_linaer_layer_call_fn_32200

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity’StatefulPartitionedCallζ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_31832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
©b
υ
__inference__traced_save_32632
file_prefixB
>savev2_jpy_model_linaer_first_dense_kernel_read_readvariableop@
<savev2_jpy_model_linaer_first_dense_bias_read_readvariableopC
?savev2_jpy_model_linaer_second_dense_kernel_read_readvariableopA
=savev2_jpy_model_linaer_second_dense_bias_read_readvariableopB
>savev2_jpy_model_linaer_third_dense_kernel_read_readvariableop@
<savev2_jpy_model_linaer_third_dense_bias_read_readvariableopC
?savev2_jpy_model_linaer_fourth_dense_kernel_read_readvariableopA
=savev2_jpy_model_linaer_fourth_dense_bias_read_readvariableopB
>savev2_jpy_model_linaer_fifth_dense_kernel_read_readvariableop@
<savev2_jpy_model_linaer_fifth_dense_bias_read_readvariableop?
;savev2_jpy_model_linaer_output_1_kernel_read_readvariableop=
9savev2_jpy_model_linaer_output_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopI
Esavev2_adam_jpy_model_linaer_first_dense_kernel_m_read_readvariableopG
Csavev2_adam_jpy_model_linaer_first_dense_bias_m_read_readvariableopJ
Fsavev2_adam_jpy_model_linaer_second_dense_kernel_m_read_readvariableopH
Dsavev2_adam_jpy_model_linaer_second_dense_bias_m_read_readvariableopI
Esavev2_adam_jpy_model_linaer_third_dense_kernel_m_read_readvariableopG
Csavev2_adam_jpy_model_linaer_third_dense_bias_m_read_readvariableopJ
Fsavev2_adam_jpy_model_linaer_fourth_dense_kernel_m_read_readvariableopH
Dsavev2_adam_jpy_model_linaer_fourth_dense_bias_m_read_readvariableopI
Esavev2_adam_jpy_model_linaer_fifth_dense_kernel_m_read_readvariableopG
Csavev2_adam_jpy_model_linaer_fifth_dense_bias_m_read_readvariableopF
Bsavev2_adam_jpy_model_linaer_output_1_kernel_m_read_readvariableopD
@savev2_adam_jpy_model_linaer_output_1_bias_m_read_readvariableopI
Esavev2_adam_jpy_model_linaer_first_dense_kernel_v_read_readvariableopG
Csavev2_adam_jpy_model_linaer_first_dense_bias_v_read_readvariableopJ
Fsavev2_adam_jpy_model_linaer_second_dense_kernel_v_read_readvariableopH
Dsavev2_adam_jpy_model_linaer_second_dense_bias_v_read_readvariableopI
Esavev2_adam_jpy_model_linaer_third_dense_kernel_v_read_readvariableopG
Csavev2_adam_jpy_model_linaer_third_dense_bias_v_read_readvariableopJ
Fsavev2_adam_jpy_model_linaer_fourth_dense_kernel_v_read_readvariableopH
Dsavev2_adam_jpy_model_linaer_fourth_dense_bias_v_read_readvariableopI
Esavev2_adam_jpy_model_linaer_fifth_dense_kernel_v_read_readvariableopG
Csavev2_adam_jpy_model_linaer_fifth_dense_bias_v_read_readvariableopF
Bsavev2_adam_jpy_model_linaer_output_1_kernel_v_read_readvariableopD
@savev2_adam_jpy_model_linaer_output_1_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ά
value¬B©.B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΙ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ³
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_jpy_model_linaer_first_dense_kernel_read_readvariableop<savev2_jpy_model_linaer_first_dense_bias_read_readvariableop?savev2_jpy_model_linaer_second_dense_kernel_read_readvariableop=savev2_jpy_model_linaer_second_dense_bias_read_readvariableop>savev2_jpy_model_linaer_third_dense_kernel_read_readvariableop<savev2_jpy_model_linaer_third_dense_bias_read_readvariableop?savev2_jpy_model_linaer_fourth_dense_kernel_read_readvariableop=savev2_jpy_model_linaer_fourth_dense_bias_read_readvariableop>savev2_jpy_model_linaer_fifth_dense_kernel_read_readvariableop<savev2_jpy_model_linaer_fifth_dense_bias_read_readvariableop;savev2_jpy_model_linaer_output_1_kernel_read_readvariableop9savev2_jpy_model_linaer_output_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopEsavev2_adam_jpy_model_linaer_first_dense_kernel_m_read_readvariableopCsavev2_adam_jpy_model_linaer_first_dense_bias_m_read_readvariableopFsavev2_adam_jpy_model_linaer_second_dense_kernel_m_read_readvariableopDsavev2_adam_jpy_model_linaer_second_dense_bias_m_read_readvariableopEsavev2_adam_jpy_model_linaer_third_dense_kernel_m_read_readvariableopCsavev2_adam_jpy_model_linaer_third_dense_bias_m_read_readvariableopFsavev2_adam_jpy_model_linaer_fourth_dense_kernel_m_read_readvariableopDsavev2_adam_jpy_model_linaer_fourth_dense_bias_m_read_readvariableopEsavev2_adam_jpy_model_linaer_fifth_dense_kernel_m_read_readvariableopCsavev2_adam_jpy_model_linaer_fifth_dense_bias_m_read_readvariableopBsavev2_adam_jpy_model_linaer_output_1_kernel_m_read_readvariableop@savev2_adam_jpy_model_linaer_output_1_bias_m_read_readvariableopEsavev2_adam_jpy_model_linaer_first_dense_kernel_v_read_readvariableopCsavev2_adam_jpy_model_linaer_first_dense_bias_v_read_readvariableopFsavev2_adam_jpy_model_linaer_second_dense_kernel_v_read_readvariableopDsavev2_adam_jpy_model_linaer_second_dense_bias_v_read_readvariableopEsavev2_adam_jpy_model_linaer_third_dense_kernel_v_read_readvariableopCsavev2_adam_jpy_model_linaer_third_dense_bias_v_read_readvariableopFsavev2_adam_jpy_model_linaer_fourth_dense_kernel_v_read_readvariableopDsavev2_adam_jpy_model_linaer_fourth_dense_bias_v_read_readvariableopEsavev2_adam_jpy_model_linaer_fifth_dense_kernel_v_read_readvariableopCsavev2_adam_jpy_model_linaer_fifth_dense_bias_v_read_readvariableopBsavev2_adam_jpy_model_linaer_output_1_kernel_v_read_readvariableop@savev2_adam_jpy_model_linaer_output_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ψ
_input_shapesζ
γ: :	::
::
::
::
::	:: : : : : : : : : :	::
::
::
::
::	::	::
::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::&("
 
_output_shapes
:
:!)

_output_shapes	
::&*"
 
_output_shapes
:
:!+

_output_shapes	
::%,!

_output_shapes
:	: -

_output_shapes
::.

_output_shapes
: 
ΪI
ο
 __inference__wrapped_model_31716
input_1N
;jpy_model_linaer_first_dense_matmul_readvariableop_resource:	K
<jpy_model_linaer_first_dense_biasadd_readvariableop_resource:	P
<jpy_model_linaer_second_dense_matmul_readvariableop_resource:
L
=jpy_model_linaer_second_dense_biasadd_readvariableop_resource:	O
;jpy_model_linaer_third_dense_matmul_readvariableop_resource:
K
<jpy_model_linaer_third_dense_biasadd_readvariableop_resource:	P
<jpy_model_linaer_fourth_dense_matmul_readvariableop_resource:
L
=jpy_model_linaer_fourth_dense_biasadd_readvariableop_resource:	O
;jpy_model_linaer_fifth_dense_matmul_readvariableop_resource:
K
<jpy_model_linaer_fifth_dense_biasadd_readvariableop_resource:	K
8jpy_model_linaer_output_1_matmul_readvariableop_resource:	G
9jpy_model_linaer_output_1_biasadd_readvariableop_resource:
identity’3jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOp’2jpy_model_linaer/fifth_dense/MatMul/ReadVariableOp’3jpy_model_linaer/first_dense/BiasAdd/ReadVariableOp’2jpy_model_linaer/first_dense/MatMul/ReadVariableOp’4jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOp’3jpy_model_linaer/fourth_dense/MatMul/ReadVariableOp’0jpy_model_linaer/output_1/BiasAdd/ReadVariableOp’/jpy_model_linaer/output_1/MatMul/ReadVariableOp’4jpy_model_linaer/second_dense/BiasAdd/ReadVariableOp’3jpy_model_linaer/second_dense/MatMul/ReadVariableOp’3jpy_model_linaer/third_dense/BiasAdd/ReadVariableOp’2jpy_model_linaer/third_dense/MatMul/ReadVariableOp―
2jpy_model_linaer/first_dense/MatMul/ReadVariableOpReadVariableOp;jpy_model_linaer_first_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0₯
#jpy_model_linaer/first_dense/MatMulMatMulinput_1:jpy_model_linaer/first_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????­
3jpy_model_linaer/first_dense/BiasAdd/ReadVariableOpReadVariableOp<jpy_model_linaer_first_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ξ
$jpy_model_linaer/first_dense/BiasAddBiasAdd-jpy_model_linaer/first_dense/MatMul:product:0;jpy_model_linaer/first_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
!jpy_model_linaer/first_dense/ReluRelu-jpy_model_linaer/first_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????²
3jpy_model_linaer/second_dense/MatMul/ReadVariableOpReadVariableOp<jpy_model_linaer_second_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ο
$jpy_model_linaer/second_dense/MatMulMatMul/jpy_model_linaer/first_dense/Relu:activations:0;jpy_model_linaer/second_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????―
4jpy_model_linaer/second_dense/BiasAdd/ReadVariableOpReadVariableOp=jpy_model_linaer_second_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ρ
%jpy_model_linaer/second_dense/BiasAddBiasAdd.jpy_model_linaer/second_dense/MatMul:product:0<jpy_model_linaer/second_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"jpy_model_linaer/second_dense/ReluRelu.jpy_model_linaer/second_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
'jpy_model_linaer/first_dropout/IdentityIdentity0jpy_model_linaer/second_dense/Relu:activations:0*
T0*(
_output_shapes
:?????????°
2jpy_model_linaer/third_dense/MatMul/ReadVariableOpReadVariableOp;jpy_model_linaer_third_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ξ
#jpy_model_linaer/third_dense/MatMulMatMul0jpy_model_linaer/first_dropout/Identity:output:0:jpy_model_linaer/third_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????­
3jpy_model_linaer/third_dense/BiasAdd/ReadVariableOpReadVariableOp<jpy_model_linaer_third_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ξ
$jpy_model_linaer/third_dense/BiasAddBiasAdd-jpy_model_linaer/third_dense/MatMul:product:0;jpy_model_linaer/third_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
!jpy_model_linaer/third_dense/ReluRelu-jpy_model_linaer/third_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????²
3jpy_model_linaer/fourth_dense/MatMul/ReadVariableOpReadVariableOp<jpy_model_linaer_fourth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ο
$jpy_model_linaer/fourth_dense/MatMulMatMul/jpy_model_linaer/third_dense/Relu:activations:0;jpy_model_linaer/fourth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????―
4jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOpReadVariableOp=jpy_model_linaer_fourth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ρ
%jpy_model_linaer/fourth_dense/BiasAddBiasAdd.jpy_model_linaer/fourth_dense/MatMul:product:0<jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"jpy_model_linaer/fourth_dense/ReluRelu.jpy_model_linaer/fourth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????°
2jpy_model_linaer/fifth_dense/MatMul/ReadVariableOpReadVariableOp;jpy_model_linaer_fifth_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ξ
#jpy_model_linaer/fifth_dense/MatMulMatMul0jpy_model_linaer/fourth_dense/Relu:activations:0:jpy_model_linaer/fifth_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????­
3jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOpReadVariableOp<jpy_model_linaer_fifth_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ξ
$jpy_model_linaer/fifth_dense/BiasAddBiasAdd-jpy_model_linaer/fifth_dense/MatMul:product:0;jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
!jpy_model_linaer/fifth_dense/ReluRelu-jpy_model_linaer/fifth_dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????©
/jpy_model_linaer/output_1/MatMul/ReadVariableOpReadVariableOp8jpy_model_linaer_output_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ζ
 jpy_model_linaer/output_1/MatMulMatMul/jpy_model_linaer/fifth_dense/Relu:activations:07jpy_model_linaer/output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¦
0jpy_model_linaer/output_1/BiasAdd/ReadVariableOpReadVariableOp9jpy_model_linaer_output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Δ
!jpy_model_linaer/output_1/BiasAddBiasAdd*jpy_model_linaer/output_1/MatMul:product:08jpy_model_linaer/output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
IdentityIdentity*jpy_model_linaer/output_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ζ
NoOpNoOp4^jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOp3^jpy_model_linaer/fifth_dense/MatMul/ReadVariableOp4^jpy_model_linaer/first_dense/BiasAdd/ReadVariableOp3^jpy_model_linaer/first_dense/MatMul/ReadVariableOp5^jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOp4^jpy_model_linaer/fourth_dense/MatMul/ReadVariableOp1^jpy_model_linaer/output_1/BiasAdd/ReadVariableOp0^jpy_model_linaer/output_1/MatMul/ReadVariableOp5^jpy_model_linaer/second_dense/BiasAdd/ReadVariableOp4^jpy_model_linaer/second_dense/MatMul/ReadVariableOp4^jpy_model_linaer/third_dense/BiasAdd/ReadVariableOp3^jpy_model_linaer/third_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2j
3jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOp3jpy_model_linaer/fifth_dense/BiasAdd/ReadVariableOp2h
2jpy_model_linaer/fifth_dense/MatMul/ReadVariableOp2jpy_model_linaer/fifth_dense/MatMul/ReadVariableOp2j
3jpy_model_linaer/first_dense/BiasAdd/ReadVariableOp3jpy_model_linaer/first_dense/BiasAdd/ReadVariableOp2h
2jpy_model_linaer/first_dense/MatMul/ReadVariableOp2jpy_model_linaer/first_dense/MatMul/ReadVariableOp2l
4jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOp4jpy_model_linaer/fourth_dense/BiasAdd/ReadVariableOp2j
3jpy_model_linaer/fourth_dense/MatMul/ReadVariableOp3jpy_model_linaer/fourth_dense/MatMul/ReadVariableOp2d
0jpy_model_linaer/output_1/BiasAdd/ReadVariableOp0jpy_model_linaer/output_1/BiasAdd/ReadVariableOp2b
/jpy_model_linaer/output_1/MatMul/ReadVariableOp/jpy_model_linaer/output_1/MatMul/ReadVariableOp2l
4jpy_model_linaer/second_dense/BiasAdd/ReadVariableOp4jpy_model_linaer/second_dense/BiasAdd/ReadVariableOp2j
3jpy_model_linaer/second_dense/MatMul/ReadVariableOp3jpy_model_linaer/second_dense/MatMul/ReadVariableOp2j
3jpy_model_linaer/third_dense/BiasAdd/ReadVariableOp3jpy_model_linaer/third_dense/BiasAdd/ReadVariableOp2h
2jpy_model_linaer/third_dense/MatMul/ReadVariableOp2jpy_model_linaer/third_dense/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
Σ

―
#__inference_signature_wrapper_32171
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_31716o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
Ζ

(__inference_output_1_layer_call_fn_32464

inputs
unknown:	
	unknown_0:
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_1_layer_call_and_return_conditional_losses_31825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ώ	
g
H__inference_first_dropout_layer_call_and_return_conditional_losses_32395

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ

ϋ
G__inference_fourth_dense_layer_call_and_return_conditional_losses_31792

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs"΅	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ιΕ
Ϊ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
first_dense
	second_dense


first_drop
third_dense
fourth_dense
fifth_dense
first_output
	optimizer

signatures"
_tf_keras_model
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
υ
"trace_0
#trace_1
$trace_2
%trace_32
0__inference_jpy_model_linaer_layer_call_fn_31859
0__inference_jpy_model_linaer_layer_call_fn_32200
0__inference_jpy_model_linaer_layer_call_fn_32229
0__inference_jpy_model_linaer_layer_call_fn_32064Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z"trace_0z#trace_1z$trace_2z%trace_3
α
&trace_0
'trace_1
(trace_2
)trace_32φ
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32275
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32328
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32099
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32134Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z&trace_0z'trace_1z(trace_2z)trace_3
ΛBΘ
 __inference__wrapped_model_31716input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
»
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ό
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator"
_tf_keras_layer
»
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Γ
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratemmmmmmmm m‘m’m£m€v₯v¦v§v¨v©vͺv«v¬v­v?v―v°"
	optimizer
,
Zserving_default"
signature_map
6:4	2#jpy_model_linaer/first_dense/kernel
0:.2!jpy_model_linaer/first_dense/bias
8:6
2$jpy_model_linaer/second_dense/kernel
1:/2"jpy_model_linaer/second_dense/bias
7:5
2#jpy_model_linaer/third_dense/kernel
0:.2!jpy_model_linaer/third_dense/bias
8:6
2$jpy_model_linaer/fourth_dense/kernel
1:/2"jpy_model_linaer/fourth_dense/bias
7:5
2#jpy_model_linaer/fifth_dense/kernel
0:.2!jpy_model_linaer/fifth_dense/bias
3:1	2 jpy_model_linaer/output_1/kernel
,:*2jpy_model_linaer/output_1/bias
 "
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B?
0__inference_jpy_model_linaer_layer_call_fn_31859input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Bώ
0__inference_jpy_model_linaer_layer_call_fn_32200inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Bώ
0__inference_jpy_model_linaer_layer_call_fn_32229inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B?
0__inference_jpy_model_linaer_layer_call_fn_32064input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32275inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32328inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32099input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32134input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ο
btrace_02?
+__inference_first_dense_layer_call_fn_32337’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zbtrace_0

ctrace_02ν
F__inference_first_dense_layer_call_and_return_conditional_losses_32348’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zctrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
π
itrace_02Σ
,__inference_second_dense_layer_call_fn_32357’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zitrace_0

jtrace_02ξ
G__inference_second_dense_layer_call_and_return_conditional_losses_32368’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zjtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Λ
ptrace_0
qtrace_12
-__inference_first_dropout_layer_call_fn_32373
-__inference_first_dropout_layer_call_fn_32378³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zptrace_0zqtrace_1

rtrace_0
strace_12Κ
H__inference_first_dropout_layer_call_and_return_conditional_losses_32383
H__inference_first_dropout_layer_call_and_return_conditional_losses_32395³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zrtrace_0zstrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ο
ytrace_02?
+__inference_third_dense_layer_call_fn_32404’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zytrace_0

ztrace_02ν
F__inference_third_dense_layer_call_and_return_conditional_losses_32415’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ς
trace_02Σ
,__inference_fourth_dense_layer_call_fn_32424’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ξ
G__inference_fourth_dense_layer_call_and_return_conditional_losses_32435’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ρ
trace_02?
+__inference_fifth_dense_layer_call_fn_32444’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ν
F__inference_fifth_dense_layer_call_and_return_conditional_losses_32455’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
ξ
trace_02Ο
(__inference_output_1_layer_call_fn_32464’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02κ
C__inference_output_1_layer_call_and_return_conditional_losses_32474’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΚBΗ
#__inference_signature_wrapper_32171input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_first_dense_layer_call_fn_32337inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_first_dense_layer_call_and_return_conditional_losses_32348inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΰBέ
,__inference_second_dense_layer_call_fn_32357inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϋBψ
G__inference_second_dense_layer_call_and_return_conditional_losses_32368inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ςBο
-__inference_first_dropout_layer_call_fn_32373inputs"³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ςBο
-__inference_first_dropout_layer_call_fn_32378inputs"³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
H__inference_first_dropout_layer_call_and_return_conditional_losses_32383inputs"³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
H__inference_first_dropout_layer_call_and_return_conditional_losses_32395inputs"³
ͺ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_third_dense_layer_call_fn_32404inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_third_dense_layer_call_and_return_conditional_losses_32415inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΰBέ
,__inference_fourth_dense_layer_call_fn_32424inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϋBψ
G__inference_fourth_dense_layer_call_and_return_conditional_losses_32435inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_fifth_dense_layer_call_fn_32444inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_fifth_dense_layer_call_and_return_conditional_losses_32455inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
άBΩ
(__inference_output_1_layer_call_fn_32464inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_output_1_layer_call_and_return_conditional_losses_32474inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
;:9	2*Adam/jpy_model_linaer/first_dense/kernel/m
5:32(Adam/jpy_model_linaer/first_dense/bias/m
=:;
2+Adam/jpy_model_linaer/second_dense/kernel/m
6:42)Adam/jpy_model_linaer/second_dense/bias/m
<::
2*Adam/jpy_model_linaer/third_dense/kernel/m
5:32(Adam/jpy_model_linaer/third_dense/bias/m
=:;
2+Adam/jpy_model_linaer/fourth_dense/kernel/m
6:42)Adam/jpy_model_linaer/fourth_dense/bias/m
<::
2*Adam/jpy_model_linaer/fifth_dense/kernel/m
5:32(Adam/jpy_model_linaer/fifth_dense/bias/m
8:6	2'Adam/jpy_model_linaer/output_1/kernel/m
1:/2%Adam/jpy_model_linaer/output_1/bias/m
;:9	2*Adam/jpy_model_linaer/first_dense/kernel/v
5:32(Adam/jpy_model_linaer/first_dense/bias/v
=:;
2+Adam/jpy_model_linaer/second_dense/kernel/v
6:42)Adam/jpy_model_linaer/second_dense/bias/v
<::
2*Adam/jpy_model_linaer/third_dense/kernel/v
5:32(Adam/jpy_model_linaer/third_dense/bias/v
=:;
2+Adam/jpy_model_linaer/fourth_dense/kernel/v
6:42)Adam/jpy_model_linaer/fourth_dense/bias/v
<::
2*Adam/jpy_model_linaer/fifth_dense/kernel/v
5:32(Adam/jpy_model_linaer/fifth_dense/bias/v
8:6	2'Adam/jpy_model_linaer/output_1/kernel/v
1:/2%Adam/jpy_model_linaer/output_1/bias/v
 __inference__wrapped_model_31716u0’-
&’#
!
input_1?????????
ͺ "3ͺ0
.
output_1"
output_1?????????¨
F__inference_fifth_dense_layer_call_and_return_conditional_losses_32455^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
+__inference_fifth_dense_layer_call_fn_32444Q0’-
&’#
!
inputs?????????
ͺ "?????????§
F__inference_first_dense_layer_call_and_return_conditional_losses_32348]/’,
%’"
 
inputs?????????
ͺ "&’#

0?????????
 
+__inference_first_dense_layer_call_fn_32337P/’,
%’"
 
inputs?????????
ͺ "?????????ͺ
H__inference_first_dropout_layer_call_and_return_conditional_losses_32383^4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 ͺ
H__inference_first_dropout_layer_call_and_return_conditional_losses_32395^4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 
-__inference_first_dropout_layer_call_fn_32373Q4’1
*’'
!
inputs?????????
p 
ͺ "?????????
-__inference_first_dropout_layer_call_fn_32378Q4’1
*’'
!
inputs?????????
p
ͺ "?????????©
G__inference_fourth_dense_layer_call_and_return_conditional_losses_32435^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
,__inference_fourth_dense_layer_call_fn_32424Q0’-
&’#
!
inputs?????????
ͺ "?????????Ύ
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32099o8’5
.’+
!
input_1?????????
p 

 
ͺ "%’"

0?????????
 Ύ
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32134o8’5
.’+
!
input_1?????????
p

 
ͺ "%’"

0?????????
 ½
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32275n7’4
-’*
 
inputs?????????
p 

 
ͺ "%’"

0?????????
 ½
K__inference_jpy_model_linaer_layer_call_and_return_conditional_losses_32328n7’4
-’*
 
inputs?????????
p

 
ͺ "%’"

0?????????
 
0__inference_jpy_model_linaer_layer_call_fn_31859b8’5
.’+
!
input_1?????????
p 

 
ͺ "?????????
0__inference_jpy_model_linaer_layer_call_fn_32064b8’5
.’+
!
input_1?????????
p

 
ͺ "?????????
0__inference_jpy_model_linaer_layer_call_fn_32200a7’4
-’*
 
inputs?????????
p 

 
ͺ "?????????
0__inference_jpy_model_linaer_layer_call_fn_32229a7’4
-’*
 
inputs?????????
p

 
ͺ "?????????€
C__inference_output_1_layer_call_and_return_conditional_losses_32474]0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 |
(__inference_output_1_layer_call_fn_32464P0’-
&’#
!
inputs?????????
ͺ "?????????©
G__inference_second_dense_layer_call_and_return_conditional_losses_32368^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
,__inference_second_dense_layer_call_fn_32357Q0’-
&’#
!
inputs?????????
ͺ "?????????¨
#__inference_signature_wrapper_32171;’8
’ 
1ͺ.
,
input_1!
input_1?????????"3ͺ0
.
output_1"
output_1?????????¨
F__inference_third_dense_layer_call_and_return_conditional_losses_32415^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
+__inference_third_dense_layer_call_fn_32404Q0’-
&’#
!
inputs?????????
ͺ "?????????