// Unlit "distance from the camera" shader: near = white, far = black (default). It needs no camera
// depth texture, so a plain camera rendering objects with this material produces a depth map.
// Used by DepthMapCamera, which swaps it onto the scene for the moment that camera renders.
Shader "Hidden/StreamDiffusion/DepthMapView"
{
    Properties
    {
        _Near ("Near distance", Float) = 0.3
        _Far ("Far distance", Float) = 15
        _NearIsWhite ("Near is white (1/0)", Float) = 1
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        Pass
        {
            Name "DepthMapView"
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Near;
                float _Far;
                float _NearIsWhite;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; };
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float eyeDepth : TEXCOORD0;
            };

            Varyings vert(Attributes a)
            {
                Varyings o;
                float3 positionWS = TransformObjectToWorld(a.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(positionWS);
                o.eyeDepth = -TransformWorldToView(positionWS).z;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float d = saturate((i.eyeDepth - _Near) / max(_Far - _Near, 0.0001));
                float v = _NearIsWhite > 0.5 ? 1.0 - d : d;
                return half4(v, v, v, 1);
            }
            ENDHLSL
        }
    }
}
