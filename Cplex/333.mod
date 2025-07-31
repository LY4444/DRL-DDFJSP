/*********************************************
 * OPL 12.9.0.0 Model
 * Author: 31523
 * Creation Date: 2024��1��18�� at ����4:19:45
 *********************************************/

 
 //��������
int n = 3;				//number of jobs
range I = 1..n;				//index of jobs
int n_i = 3;			//operation number of job i
range J_i = 1..n_i;			//index of operations
int r_f = 3;			//number of factories
range F = 1..r_f;			//index of factory
int m_ijf = 3;			//machine number that can handdle O_ij of factory f
range K_ijf = 1..m_ijf;		//index of machines that can handdle O_ij of factory f

int AT[I] = [2, 3, 5];
int DD[I] = [10, 12, 14];
int TM = 3;
int TF = 6;
float U[I] = [1.4, 1.4];


//���ݶ�ȡ
float x_fkij[F][K_ijf][I][J_i] = ...;
float pt_fkij[F][K_ijf][I][J_i] = ...;			//time of machine MIndex in FIndex deal OpIndex of JobIndex
float bd_fkij[F][K_ijf][I][J_i] = ...;          // # #
float rt_fkij[F][K_ijf][I][J_i] = ...;          // # #
int M = 100000;

//���߱���
dvar boolean X[I][J_i][F][K_ijf];			//X==1 if O_ij is assigned to machine k of factory f
dvar boolean Y[I][J_i][I][J_i];			//Y==1 if O_ij is processed before O_i'j'
dvar boolean Z1[I][J_i];  				//һ������֮��ת��
dvar boolean Z2[I][J_i];  				//��������֮��ת��

dvar float B[I][J_i];					//defines the starting time of O_ij
dvar float C[I];
dvar float MT[I];						//makespan
dvar float MTJ;

//�Ż�Ŀ��
minimize MTJ;

//Լ������
subject to{

MTJ<=100000;

//cons02:ͬ������ֻ����һ̨�����ϼӹ�
forall(i in I, j in J_i) sum(f in F, k in K_ijf) X[i][j][f][k] <= 1;
forall (i in I, j in J_i) {
	sum (f in F, k in K_ijf) X[i][j][f][k] * x_fkij[f][k][i][j] == 1;
	}

//cons02:ֻ����һ��ת�Ʒ�ʽ
forall (i in I, j in 2..n_i) {
	Z1[i][j] + Z2[i][j] <= 1 - sum(f in F) sum(k in K_ijf) X[i][j][f][k] * X[i][j-1][f][k];
	Z1[i][j] >= sum(f in F) (sum(k in K_ijf,k1 in K_ijf:k!=k1) X[i][j][f][k] * X[i][j-1][f][k1]);
	Z2[i][j] >= 1 - sum(f in F) (sum(k in K_ijf) X[i][j][f][k] * sum(k1 in K_ijf) X[i][j-1][f][k1]);

	}

//cons03:��һ��������Ҫת��
forall (i in I) {
	Z1[i][1] + Z2[i][1] == 0;
	}

//cons04:����ʼ�ӹ�ʱ���Լ��
forall (i in I) {
	B[i][1] >= AT[i];
	}

//cons03:ͬһ�������Ⱥ���Լ������ǰһ������깤ʱ���ں�һ����Ŀ�ʼʱ��֮ǰ
forall (i in I, j in 1..n_i-1) {
	B[i][j+1] >= B[i][j] + sum(f in F, k in K_ijf) (X[i][j][f][k] * (pt_fkij[f][k][i][j] + rt_fkij[f][k][i][j] * bd_fkij[f][k][i][j])) + Z1[i][j+1] * TM + Z2[i][j+1] * TF;
	}

//cons04\05:��֤��ҵ���䲻�ص�
forall (i1 in I, i2 in I, j1 in J_i, j2 in J_i, f in F, k in K_ijf:j1 != j2){
	B[i1][j1] + X[i1][j1][f][k] * pt_fkij[f][k][i1][j1] <= B[i2][j2] + M * (3-Y[i1][j1][i2][j2] - X[i1][j1][f][k] - X[i2][j2][f][k]);
	B[i2][j2] + X[i2][j2][f][k] * pt_fkij[f][k][i2][j2] <= B[i1][j1] + M * (2+Y[i1][j1][i2][j2] - X[i1][j1][f][k] - X[i2][j2][f][k]);
	}

//cons06:�깤ʱ��ļ��㷽��
forall (i in I) {
	C[i] >= B[i][n_i] + sum(f in F, k in K_ijf) X[i][n_i][f][k] * (pt_fkij[f][k][i][n_i] + rt_fkij[f][k][i][n_i] * bd_fkij[f][k][i][n_i]);
	}

//cons07:��ʼʱ��
forall (i in I, j in J_i) {
	B[i][j] >= 0;
	} 

//����ʱ��ļ��㷽��
forall (i in I) {
	MT[i] >= C[i] - DD[i];
	MT[i] >= 0;
}

MTJ >= sum(i in I) (MT[i] * U[i]) / n;

}

execute
{
writeln("��Сƽ������ʱ��:",MTJ);
}