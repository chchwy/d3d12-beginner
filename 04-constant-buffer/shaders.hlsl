
/*
cbuffer cb : register(b0)
{
	float4x4 rotation;
}
*/

struct Interpolator
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

struct Output
{
	float4 color : SV_TARGET0;
};

Interpolator VSMain(float3 position : POSITION, float4 color : COLOR)
{
	Interpolator o;

	o.position = float4(position, 1); //mul(float4(position, 1), rotation);
	o.color = color;

	return o;
}

Output PSMain(Interpolator i) : SV_TARGET
{
	Output o;
	o.color = i.color;
	return o;
}
