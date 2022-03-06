A = sym('a', [2 3]);
B = sym('b', [2 3]);
C = sym('c', [2 3]);


%Assemble the system and multiply
D1=blkdiag(A,B,C);
D = sym('d', [9 3]);
D2=D1*D

%Multiply individually and assemble
V=size(A)
D3=[A*D(1:3,:);
B*D(4:6,:);
C*D(7:9,:);]


A=rand([10,6]);
B=rand([6,6]);

%If I have
AB=A*B
AB2=[A(1:5,:)*B;
    A(6:10,:)*B]


A = sym('a', [3 3]);
B = sym('b', [3 3]);

%A=[1,0,0;3,0,1;1,1,0]
%B=[0,1,0;1,1,0;0,2,1]

[eigenVectors, eigenValues] = eig(B, A);
