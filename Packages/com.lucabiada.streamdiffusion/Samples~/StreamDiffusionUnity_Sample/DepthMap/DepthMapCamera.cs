using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace StreamDiffusion.Samples
{
    /// <summary>
    /// A depth-map camera: renders what it sees as a grey "distance" picture (near = white, far = black)
    /// into <see cref="target"/>, the format depth ControlNets expect. For the moment this camera renders,
    /// the scene's renderers (those in its culling mask) are given a distance material, then restored, so
    /// the other cameras are not affected.
    ///
    /// Use: put it on a camera that sees the same scene as the model's source camera (a child of it keeps
    /// the same view), assign <see cref="target"/> (a RenderTexture, e.g. 512x512), then assign that texture
    /// as the DiffusionController's Control Texture with the Texture control source and the "passthrough"
    /// preprocessor.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public sealed class DepthMapCamera : MonoBehaviour
    {
        [Tooltip("Receives the depth picture.")]
        public RenderTexture target;
        [Tooltip("Distance (metres) that maps to the nearest value.")]
        public float near = 0.3f;
        [Tooltip("Distance (metres) that maps to the farthest value. Anything beyond is background (black).")]
        public float far = 30f;
        [Tooltip("On: near objects are white, far are black (what depth ControlNets are trained on).")]
        public bool nearIsWhite = true;
        [Tooltip("Seconds between looks for new / removed renderers in the scene.")]
        public float rescanInterval = 0.5f;
        public Shader shader;

        struct Saved
        {
            public Renderer renderer;
            public Material[] materials;
        }

        Camera _cam;
        Material _mat;
        readonly List<Renderer> _renderers = new List<Renderer>();
        readonly List<Saved> _saved = new List<Saved>();
        float _nextScan;

        void OnEnable()
        {
            _cam = GetComponent<Camera>();
            if (shader == null) shader = Shader.Find("Hidden/StreamDiffusion/DepthMapView");
            if (shader != null) _mat = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            else Debug.LogWarning("[StreamDiffusion] DepthMapCamera: shader 'Hidden/StreamDiffusion/DepthMapView' not found");

            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = nearIsWhite ? Color.black : Color.white;
            _cam.allowHDR = false;
            _cam.allowMSAA = false;
            if (target != null) _cam.targetTexture = target;

            RenderPipelineManager.beginCameraRendering += OnBegin;
            RenderPipelineManager.endCameraRendering += OnEnd;
        }

        void OnDisable()
        {
            RenderPipelineManager.beginCameraRendering -= OnBegin;
            RenderPipelineManager.endCameraRendering -= OnEnd;
            Restore();
            if (_cam != null && _cam.targetTexture == target) _cam.targetTexture = null;
            if (_mat != null) Destroy(_mat);
        }

        void Rescan()
        {
            _renderers.Clear();
            foreach (var r in FindObjectsByType<Renderer>(FindObjectsInactive.Exclude))
            {
                if (r == null || !r.enabled) continue;
                if (r is ParticleSystemRenderer || r is TrailRenderer || r is LineRenderer) continue;
                if ((_cam.cullingMask & (1 << r.gameObject.layer)) == 0) continue;
                _renderers.Add(r);
            }
        }

        void OnBegin(ScriptableRenderContext context, Camera cam)
        {
            if (cam != _cam || _mat == null) return;
            if (target != null && _cam.targetTexture != target) _cam.targetTexture = target;
            if (Time.unscaledTime >= _nextScan || _renderers.Count == 0)
            {
                Rescan();
                _nextScan = Time.unscaledTime + Mathf.Max(0.1f, rescanInterval);
            }
            _mat.SetFloat("_Near", near);
            _mat.SetFloat("_Far", far);
            _mat.SetFloat("_NearIsWhite", nearIsWhite ? 1f : 0f);
            _cam.backgroundColor = nearIsWhite ? Color.black : Color.white;

            Restore();
            foreach (var r in _renderers)
            {
                if (r == null) continue;
                var mats = r.sharedMaterials;
                _saved.Add(new Saved { renderer = r, materials = mats });
                var swapped = new Material[mats.Length];
                for (int i = 0; i < swapped.Length; i++) swapped[i] = _mat;
                r.sharedMaterials = swapped;
            }
        }

        void OnEnd(ScriptableRenderContext context, Camera cam)
        {
            if (cam != _cam) return;
            Restore();
        }

        void Restore()
        {
            for (int i = 0; i < _saved.Count; i++)
            {
                var s = _saved[i];
                if (s.renderer != null) s.renderer.sharedMaterials = s.materials;
            }
            _saved.Clear();
        }
    }
}
