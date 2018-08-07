// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// Copyright (c) 2010 libmv contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "essentialF10Solver.hpp"

#include <iostream>

namespace aliceVision {

	int F10RelativePose(const Mat &X, const Mat &U, std::vector<Mat3> *F, std::vector<Mat21> *L) {
		eigen_assert((X.rows() == 10 && X.cols() == 2) && "The first parameter (x) must be a 10x2 matrix");
		eigen_assert((U.rows() == 10 && U.cols() == 2) && "The second parameter (u) must be a 10x2 matrix");

		Eigen::Matrix<double, 10, 1> Z1;
		Eigen::Matrix<double, 10, 1> Z2;
		Eigen::Matrix<double, 10, 16> A;

		Z1.array() = X.col(0).array() * X.col(0).array() + X.col(1).array() * X.col(1).array();
		Z2.array() = U.col(0).array() * U.col(0).array() + U.col(1).array() * U.col(1).array();

		A.col(0).array() = X.col(0).array() * U.col(0).array();
		A.col(1).array() = X.col(0).array() * U.col(1).array();
		A.col(2).array() = X.col(1).array() * U.col(0).array();
		A.col(3).array() = X.col(1).array() * U.col(1).array();
		A.col(4).array() = U.col(0).array() * Z1.array();
		A.col(5).array() = U.col(0).array();
		A.col(6).array() = U.col(1).array() * Z1.array();
		A.col(7).array() = U.col(1).array();
		A.col(8).array() = X.col(0).array() * Z2.array();
		A.col(9).array() = X.col(0).array();
		A.col(10).array() = X.col(1).array() * Z2.array();
		A.col(11).array() = X.col(1).array();
		A.col(12).array() = Z1.array() * Z2.array();
		A.col(13).array() = Z1.array();
		A.col(14).array() = Z2.array();
		A.col(15).fill(1.0);

		Eigen::Matrix<double, 10, 6> Mr = A.block<10, 10>(0, 0).lu().solve(A.block<10, 6>(0, 10));

		Eigen::Matrix<double, 29, 1> params;

		params << Mr(5, 0), Mr(5, 1), -Mr(4, 0), -Mr(4, 1), Mr(5, 2), Mr(5, 3), Mr(5, 4) - Mr(4, 2),
			Mr(5, 5) - Mr(4, 3), -Mr(4, 4), -Mr(4, 5),
			Mr(7, 0), Mr(7, 1), -Mr(6, 0), -Mr(6, 1), Mr(7, 2), Mr(7, 3), Mr(7, 4) - Mr(6, 2),
			Mr(7, 5) - Mr(6, 3), -Mr(6, 4), -Mr(6, 5),
			Mr(9, 0), Mr(9, 1) - Mr(8, 0), -Mr(8, 1), Mr(9, 2), Mr(9, 4), Mr(9, 3) - Mr(8, 2),
			-Mr(8, 3), Mr(9, 5) - Mr(8, 4), -Mr(8, 5);

		Mat Ls(2, 10);
		int nsols = f10e_gb(params, Ls);

		if (nsols > 0)
		{
			Eigen::Matrix<double, 4, 1> m1;
			Eigen::Matrix<double, 6, 1> m2;
			Eigen::Matrix<double, 6, 1> m3;
			Eigen::Matrix<double, 10, 1> b;

			b << Mr(5, 0), Mr(5, 1), -Mr(4, 0), -Mr(4, 1), Mr(5, 2), Mr(5, 3),
				Mr(5, 4) - Mr(4, 2), Mr(5, 5) - Mr(4, 3), -Mr(4, 4), -Mr(4, 5);

			F->resize(nsols);
			L->resize(nsols);
			for (int i = 0; i < nsols; i++)
			{
				double l1 = Ls(0, i);
				double l2 = Ls(1, i);
				double l1l1 = l1 * l1;
				double l1l2 = l1 * l2;
				double f23;

				m1 << l1l2, l1, l2, 1;
				m2 << l1l2 * l1, l1l1, l1l2, l1, l2, 1;
				f23 = -b.block<6, 1>(4, 0).dot(m2) / b.block<4, 1>(0, 0).dot(m1);
				m3 << l2 * f23, f23, l1l2, l1, l2, 1;

				(*L)[i] <<	l1, l2;
				(*F)[i] <<	m3.dot(-Mr.row(0)), m3.dot(-Mr.row(1)), m3.dot(-Mr.row(9)),
							m3.dot(-Mr.row(2)), m3.dot(-Mr.row(3)), f23,
							m3.dot(-Mr.row(5)), m3.dot(-Mr.row(7)), 1;

				/*Fs(0, i) = m3.dot(-Mr.row(0));
				Fs(1, i) = m3.dot(-Mr.row(2));
				Fs(2, i) = m3.dot(-Mr.row(5));
				Fs(3, i) = m3.dot(-Mr.row(1));
				Fs(4, i) = m3.dot(-Mr.row(3));
				Fs(5, i) = m3.dot(-Mr.row(7));
				Fs(6, i) = m3.dot(-Mr.row(9));
				Fs(7, i) = f23;
				Fs(8, i) = 1;*/
			}
		}

		return nsols;




	}






} // namespace aliceVision
