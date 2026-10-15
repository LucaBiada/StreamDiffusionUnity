using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// Animates every child of this object: each one scales uniformly between
    /// <see cref="minScale"/> and <see cref="maxScale"/>, taking its own random amount of time
    /// (within <see cref="durationRange"/>) to go from one end to the other, and back again.
    /// Each child also starts at a random point in its cycle so they never move in sync.
    /// </summary>
    [AddComponentMenu("StreamDiffusion/Sphere Scale Animator (sample)")]
    public sealed class SphereScaleAnimator : MonoBehaviour
    {
        [Min(0.001f)] public float minScale = 0.1f;
        [Min(0.001f)] public float maxScale = 2f;

        [Tooltip("Each child gets a random duration in this range (seconds) for one min->max sweep.")]
        public Vector2 durationRange = new Vector2(1.5f, 8f);

        [Tooltip("Ease the ends of each sweep instead of a constant-speed ramp.")]
        public bool smooth = true;

        [Tooltip("0 = pick a new random seed every run, otherwise a fixed seed gives the same animation each time.")]
        public int seed = 0;

        Transform[] _items;
        float[] _duration;   // seconds for one min->max sweep
        float[] _offset;     // seconds into the cycle at t = 0

        void Start() => Randomize();

        /// <summary>Re-read the children and give each a new random duration and phase.</summary>
        [ContextMenu("Re-randomize")]
        public void Randomize()
        {
            var rng = seed != 0 ? new System.Random(seed) : new System.Random();
            int n = transform.childCount;
            _items = new Transform[n];
            _duration = new float[n];
            _offset = new float[n];

            float lo = Mathf.Min(durationRange.x, durationRange.y);
            float hi = Mathf.Max(durationRange.x, durationRange.y);
            lo = Mathf.Max(0.05f, lo);
            hi = Mathf.Max(lo, hi);

            for (int i = 0; i < n; i++)
            {
                _items[i] = transform.GetChild(i);
                _duration[i] = lo + (float)rng.NextDouble() * (hi - lo);
                // a full cycle (up and back) is 2 * duration; start anywhere inside it
                _offset[i] = (float)rng.NextDouble() * 2f * _duration[i];
            }
        }

        void Update()
        {
            if (_items == null) return;
            float time = Time.time;
            for (int i = 0; i < _items.Length; i++)
            {
                var t = _items[i];
                if (t == null) continue;
                // 0 -> 1 over `duration`, then 1 -> 0 over `duration`
                float k = Mathf.PingPong((time + _offset[i]) / _duration[i], 1f);
                if (smooth) k = k * k * (3f - 2f * k);
                float s = Mathf.Lerp(minScale, maxScale, k);
                t.localScale = new Vector3(s, s, s);
            }
        }
    }
}
