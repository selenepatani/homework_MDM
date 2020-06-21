function rhs=rhs_vdp(t,x,dummy,mu)
rhs=[(mu(1)-mu(2)*x(2))*x(1); (mu(3)*x(1)-mu(4))*x(2)];