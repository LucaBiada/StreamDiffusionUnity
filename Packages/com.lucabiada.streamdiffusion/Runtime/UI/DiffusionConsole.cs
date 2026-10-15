using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UIElements;

namespace StreamDiffusion
{
    /// <summary>
    /// The in-app console (mimics the Unity Editor console: levels, filters, search, details) plus
    /// an activity strip that shows what a long operation is doing, how far along it is, and
    /// whether the server is still alive. Fed by the controller log and by Unity's own log, so
    /// it works the same in the Editor and in an exported player.
    /// </summary>
    public sealed class DiffusionConsole
    {
        struct Entry
        {
            public DateTime time;
            public int level;          // 0 info, 1 warning, 2 error
            public string source;      // "app", "server", "unity"
            public string text;
            public string stack;
        }

        const int MaxEntries = 3000;
        static readonly Color Muted = new Color(168 / 255f, 168 / 255f, 168 / 255f, 1f);
        static readonly Color Warn = new Color32(245, 180, 60, 255);
        static readonly Color Err = new Color32(240, 100, 100, 255);
        static readonly Color Accent = new Color32(55, 142, 240, 255);

        readonly DiffusionController _c;
        readonly ConcurrentQueue<Entry> _q = new ConcurrentQueue<Entry>();
        readonly List<Entry> _all = new List<Entry>();
        readonly List<Entry> _view = new List<Entry>();

        bool _open;

        // UI
        VisualElement _dock, _strip, _barFill, _barTrack, _head;
        ListView _list;
        Label _stripTitle, _stripDetail, _stripRight, _headLine;
        VisualElement _arrow;
        float _installFraction = -1f;
        string _installMessage;

        public VisualElement Dock => _dock;
        public VisualElement Strip => _strip;

        public DiffusionConsole(DiffusionController c, bool expanded)
        {
            _c = c;
            _open = expanded;
            c.Log += OnAppLog;
            Application.logMessageReceivedThreaded += OnUnityLog;
            BuildStrip();
            BuildDock();
            ApplyOpen();
        }

        public void Dispose()
        {
            if (_c != null) _c.Log -= OnAppLog;
            Application.logMessageReceivedThreaded -= OnUnityLog;
        }

        // ------------------------------------------------------------------ feed (any thread)
        void OnAppLog(string line)
        {
            if (line == null) return;
            bool server = line.StartsWith("[server]");
            string text = server ? line.Substring(8).TrimStart() : line;
            int level = 0;
            if (server)
            {
                if (text.Contains(" ERROR ") || text.Contains("Traceback") || text.Contains("CRITICAL") || text.StartsWith("[E]")) level = 2;
                else if (text.Contains(" WARNING ") || text.Contains("Warning") || text.StartsWith("[W]")) level = 1;
            }
            else if (text.StartsWith("server error") || text.Contains("[Error]")) level = 2;
            else if (text.StartsWith("warning")) level = 1;
            Enqueue(level, server ? "server" : "app", text, null);
        }

        void OnUnityLog(string condition, string stack, LogType type)
        {
            if (condition != null && condition.StartsWith("[SD] ")) return;   // already shown from the controller log
            int level = type == LogType.Warning ? 1 : (type == LogType.Log ? 0 : 2);
            Enqueue(level, "unity", condition ?? "", stack);
        }

        void Enqueue(int level, string source, string text, string stack)
        {
            // progress bars (tqdm) redraw the same line many times; keep one
            if (text.Length > 600) text = text.Substring(0, 600) + " ...";
            _q.Enqueue(new Entry { time = DateTime.Now, level = level, source = source, text = text, stack = stack });
        }

        // ------------------------------------------------------------------ UI
        void BuildStrip()
        {
            _strip = new VisualElement();
            _strip.style.display = DisplayStyle.None;
            _strip.AddToClassList("sd-solid");
            _strip.style.borderBottomWidth = 1;
            _strip.style.borderBottomColor = (Color)new Color32(42, 42, 42, 255);
            _strip.style.paddingLeft = 24; _strip.style.paddingRight = 24;
            _strip.style.paddingTop = 8; _strip.style.paddingBottom = 8;

            var row = new VisualElement { style = { flexDirection = FlexDirection.Row, alignItems = Align.Center } };
            _stripTitle = new Label("");
            _stripTitle.AddToClassList("ds-body-2");
            _stripTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
            _stripTitle.style.marginRight = 12;
            _stripDetail = new Label("");
            _stripDetail.AddToClassList("ds-caption");
            _stripDetail.style.color = Muted;
            _stripDetail.style.flexGrow = 1;
            _stripRight = new Label("");
            _stripRight.AddToClassList("ds-caption");
            _stripRight.style.color = Muted;
            row.Add(_stripTitle); row.Add(_stripDetail); row.Add(_stripRight);
            _strip.Add(row);

            _barTrack = new VisualElement();
            _barTrack.style.height = 6;
            _barTrack.style.marginTop = 6;
            _barTrack.style.backgroundColor = (Color)new Color32(42, 42, 42, 255);
            _barTrack.style.borderTopLeftRadius = 3; _barTrack.style.borderTopRightRadius = 3;
            _barTrack.style.borderBottomLeftRadius = 3; _barTrack.style.borderBottomRightRadius = 3;
            _barFill = new VisualElement();
            _barFill.style.height = 6;
            _barFill.style.backgroundColor = Accent;
            _barFill.style.borderTopLeftRadius = 3; _barFill.style.borderTopRightRadius = 3;
            _barFill.style.borderBottomLeftRadius = 3; _barFill.style.borderBottomRightRadius = 3;
            _barTrack.Add(_barFill);
            _strip.Add(_barTrack);
        }

        void BuildDock()
        {
            _dock = new VisualElement();
            _dock.style.flexShrink = 0;
            _dock.AddToClassList("sd-solid");
            _dock.style.borderTopWidth = 1;
            _dock.style.borderTopColor = (Color)new Color32(42, 42, 42, 255);

            // one line: the newest message; click it to open or close the full list
            _head = new VisualElement { style = { flexDirection = FlexDirection.Row, alignItems = Align.Center,
                paddingLeft = 16, paddingRight = 16, height = 28 } };
            // drawn (not a font glyph): an equilateral triangle, up = "open the list", down = "close it"
            _arrow = new VisualElement();
            _arrow.style.width = 18;
            _arrow.style.height = 16;
            _arrow.style.marginRight = 8;
            _arrow.generateVisualContent += ctx =>
            {
                var r = _arrow.contentRect;
                float side = Mathf.Min(r.width, r.height / 0.866f) * 0.65f;
                float h = side * 0.866f;
                float cx = r.width * 0.5f, cy = r.height * 0.5f;
                var p = ctx.painter2D;
                p.fillColor = Accent;
                p.BeginPath();
                if (_open)   // pointing down
                {
                    p.MoveTo(new Vector2(cx - side / 2f, cy - h / 2f));
                    p.LineTo(new Vector2(cx + side / 2f, cy - h / 2f));
                    p.LineTo(new Vector2(cx, cy + h / 2f));
                }
                else         // pointing up
                {
                    p.MoveTo(new Vector2(cx - side / 2f, cy + h / 2f));
                    p.LineTo(new Vector2(cx + side / 2f, cy + h / 2f));
                    p.LineTo(new Vector2(cx, cy - h / 2f));
                }
                p.ClosePath();
                p.Fill();
            };
            _headLine = new Label("");
            _headLine.style.flexGrow = 1;
            _headLine.style.flexShrink = 1;
            _headLine.style.fontSize = 12;
            _headLine.style.whiteSpace = WhiteSpace.NoWrap;
            _headLine.style.overflow = Overflow.Hidden;
            _head.style.cursor = new StyleCursor(StyleKeyword.Initial);
            _head.Add(_arrow); _head.Add(_headLine);
            _head.RegisterCallback<ClickEvent>(_ => Toggle());
            _dock.Add(_head);

            _list = new ListView
            {
                fixedItemHeight = 20,
                virtualizationMethod = CollectionVirtualizationMethod.FixedHeight,
                selectionType = SelectionType.None,
                showAlternatingRowBackgrounds = AlternatingRowBackground.None,
                showBorder = false,
                itemsSource = _view,
            };
            _list.style.height = 220;
            _list.makeItem = () =>
            {
                var l = new Label();
                l.style.unityTextAlign = TextAnchor.MiddleLeft;
                l.style.paddingLeft = 42;   // 16 (row padding) + 18 (arrow) + 8 (gap): same left edge as the first line
                l.style.whiteSpace = WhiteSpace.NoWrap;
                l.style.overflow = Overflow.Hidden;
                l.style.fontSize = 12;
                return l;
            };
            _list.bindItem = (e, i) =>
            {
                var l = (Label)e;
                if (i < 0 || i >= _view.Count) return;
                var en = _view[i];
                l.text = en.time.ToString("HH:mm:ss") + "  " + FirstLine(en.text);
                l.tooltip = en.text;
                l.style.backgroundColor = (i & 1) == 1 ? (Color)new Color32(16, 16, 16, 255) : new Color(0f, 0f, 0f, 0f);
                l.style.color = Tint(en);
            };
            _dock.Add(_list);

            // no "List is empty" placeholder
            _list.schedule.Execute(() =>
            {
                foreach (var e in _list.Query<Label>().ToList())
                    if (e.ClassListContains("unity-list-view__empty-label") || e.ClassListContains("unity-base-list-view__empty-label"))
                        e.style.display = DisplayStyle.None;
            }).Every(500);
        }

        static Color Tint(Entry en) =>
            en.level == 2 ? Err : en.level == 1 ? Warn : (en.source == "server" ? Muted : Color.white);

        void Toggle()
        {
            _open = !_open;
            ApplyOpen();
        }

        void ApplyOpen()
        {
            _list.style.display = _open ? DisplayStyle.Flex : DisplayStyle.None;
            _arrow.MarkDirtyRepaint();
            if (_open && _view.Count > 0) _list.ScrollToItem(-1);
        }

        static string FirstLine(string t)
        {
            int i = t.IndexOf('\n');
            return i < 0 ? t : t.Substring(0, i);
        }

        // ------------------------------------------------------------------ per frame
        public void SetInstallProgress(float fraction, string message)
        {
            _installFraction = fraction; _installMessage = message;
        }

        public void Update()
        {
            bool got = false;
            while (_q.TryDequeue(out var e))
            {
                // coalesce progress-bar redraws (tqdm / Loading ... NN%|) into one line
                if (_all.Count > 0 && e.source == "server" && e.text.Contains("%|"))
                {
                    var last = _all[_all.Count - 1];
                    if (last.source == "server" && last.text.Contains("%|") && Prefix(last.text) == Prefix(e.text))
                    {
                        _all[_all.Count - 1] = e;
                        got = true;
                        continue;
                    }
                }
                _all.Add(e);
                                if (_all.Count > MaxEntries) _all.RemoveAt(0);
                got = true;
            }
            if (got)
            {
                _view.Clear();
                _view.AddRange(_all);
                _list.RefreshItems();
                if (_open && _view.Count > 0) _list.ScrollToItem(-1);
                if (_all.Count > 0)
                {
                    var last = _all[_all.Count - 1];
                    _headLine.text = last.time.ToString("HH:mm:ss") + "  " + FirstLine(last.text);
                    _headLine.style.color = Tint(last);
                }
            }
            UpdateStrip();
        }

        static string Prefix(string t)
        {
            int i = t.IndexOf("%|", StringComparison.Ordinal);
            int j = Math.Max(0, i - 8);
            return t.Substring(0, j);
        }

        static string Clock(float seconds)
        {
            int s = (int)seconds;
            return (s / 60).ToString("0") + ":" + (s % 60).ToString("00");
        }

        void UpdateStrip()
        {
            var snap = _c.Activity.Get();
            bool installing = _c.Phase == DiffusionPhase.Installing;
            if (!snap.active && !installing)
            {
                _strip.style.display = DisplayStyle.None;
                return;
            }
            _strip.style.display = DisplayStyle.Flex;

            if (installing && !snap.active)
            {
                _stripTitle.text = "Installing the environment";
                _stripDetail.text = _installMessage ?? "";
                _stripRight.text = "";
                SetBar(_installFraction, false);
                return;
            }

            _stripTitle.text = snap.title;
            _stripDetail.text = snap.detail;
            var sb = new StringBuilder();
            sb.Append(Clock(snap.elapsed)).Append(" elapsed");
            if (snap.fraction >= 0f) sb.Append("  -  ").Append(snap.estimated ? "~" : "").Append((snap.fraction * 100f).ToString("0")).Append(" %");
            if (snap.gpuUtil >= 0f) sb.Append("  -  GPU ").Append(snap.gpuUtil.ToString("0")).Append(" %, ").Append(snap.gpuMemGb.ToString("0.0")).Append(" GB");
            bool quietGpu = snap.gpuUtil >= 0f && snap.gpuUtil < 5f;
            bool stalled = snap.sinceOutput > 90f && quietGpu;
            if (snap.sinceOutput > 5f) sb.Append("  -  last output ").Append(snap.sinceOutput.ToString("0")).Append(" s ago");
            if (stalled) sb.Append("  -  no GPU activity and no output: may be stalled");
            _stripRight.text = sb.ToString();
            _stripRight.style.color = stalled ? Warn : Muted;
            SetBar(snap.fraction, snap.fraction < 0f);
        }

        void SetBar(float fraction, bool indeterminate)
        {
            if (indeterminate)
            {
                // a block sweeping back and forth: "alive, amount unknown"
                float t = Mathf.PingPong(Time.unscaledTime * 0.6f, 1f);
                _barFill.style.marginLeft = Length.Percent(t * 80f);
                _barFill.style.width = Length.Percent(20f);
            }
            else
            {
                _barFill.style.marginLeft = 0;
                _barFill.style.width = Length.Percent(Mathf.Clamp01(fraction) * 100f);
            }
        }
    }
}
