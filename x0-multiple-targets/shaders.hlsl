//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

struct Interpolator
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

struct Output
{
	float4 rgba : SV_TARGET0;
	float4 debug01 : SV_TARGET1;
};

Interpolator VSMain(float4 position : POSITION, float4 color : COLOR)
{
	Interpolator o;

	o.position = position;
	o.color = color;

	return o;
}

Output PSMain(Interpolator i) : SV_TARGET
{
	Output o;
	o.rgba = i.color;
	o.debug01 = i.position;
	return o;
}
