function H=homography_solve(u,v)
N=size(u,2);
%initialization of A and b
A=zeros(2*N,8);
b=zeros(2*N,1);
for i=1:N
	b(2*i-1)=v(1,i);
	b(2*i)=v(2,i);
end
for j=1:N
	A(2*j-1,1)=u(1,j);
	A(2*j-1,2)=u(2,j);
	A(2*j-1,3)=1;
	A(2*j-1,7)=-u(1,j)*v(1,j);
	A(2*j-1,8)=-u(2,j)*v(1,j);
	A(2*j,4)=u(1,j);
	A(2*j,5)=u(2,j);
	A(2*j,6)=1;
	A(2*j,7)=-u(1,j)*v(2,j);
	A(2*j,8)=-u(2,j)*v(2,j);
end
x=(A'*A)\A'*b;
H=[x(1),x(2),x(3);x(4),x(5),x(6);x(7),x(8),1];
end