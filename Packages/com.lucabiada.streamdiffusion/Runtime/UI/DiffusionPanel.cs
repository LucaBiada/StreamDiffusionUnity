using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using DesignSystem.Runtime.Behaviour;
using UnityEngine;
using UnityEngine.UIElements;

namespace StreamDiffusion.UI
{
    public enum DiffusionPanelPage { Stream, Performance, Model, Fx, Ndi, Presets, Environment, About }

    /// <summary>
    /// Runtime control panel in the design-system showcase style, organised in pages like the
    /// StreamDiffusionTD operator: Stream, Performance, Model, Presets, Environment, About. The output preview stays visible next to the pages. Every control edits a public
    /// field on DiffusionController, so the inspector and the UI are always in agreement.
    /// </summary>
    [RequireComponent(typeof(DiffusionController))]
    [AddComponentMenu("StreamDiffusion/Diffusion Panel")]
    public sealed class DiffusionPanel : MonoBehaviour
    {
        [Header("Pages (switch off for a smaller interface; remove this component for none)")]
        public bool showStream = true;
        public bool showPerformance = true;
        public bool showModel = true;
        public bool showNdi = true;
        public bool showFx = true;
        public bool showPresets = true;
        public bool showEnvironment = true;
        public bool showAbout = true;
        public DiffusionPanelPage startPage = DiffusionPanelPage.Stream;

        [Header("Chrome")]
        public bool showHeader = true;
        [Tooltip("Text shown in the header bar.")]
        public string titleText = "StreamDiffusionUnity";
        [Tooltip("The output preview + Start/Stop card on the right.")]
        public bool showOutput = true;
        [Tooltip("Show the image being sent to the model above the result, so input and output can be compared.")]
        public bool showInputPreview = true;
        [Tooltip("Show the small SD button that hides/shows the whole panel.")]
        public bool showHideButton = true;
        public bool startHidden = false;
        [Tooltip("Show the console at the bottom (Unity-console style) and the activity strip with progress.")]
        public bool showConsole = true;
        [Tooltip("Open the full message list at start (otherwise a single line showing the latest message; click it to open).")]
        public bool openConsoleAtStart = false;

        static readonly Color Muted = new Color(168 / 255f, 168 / 255f, 168 / 255f, 1f);

        readonly List<Action> _binders = new List<Action>();
        readonly List<(DiffusionPanelPage page, Button tab, VisualElement body)> _pages =
            new List<(DiffusionPanelPage, Button, VisualElement)>();
        DiffusionController _c;
        UIDocument _doc;
        VisualElement _hideTarget;
        DiffusionConsole _console;

        Button _startBtn, _installBtn;
        Label _statusBadge, _status;
        Label _envState, _logBox, _serverLog;
        ProgressBar _installBar;
        Image _preview, _inputPreview;
        VisualElement _vramFill;
        Image _controlPreview, _referencePreview, _controlMapPreview;
        Label _controlPreviewCap, _referencePreviewCap, _controlMapCap;
        Label _vramLabel;
        readonly Dictionary<string, Label> _stat = new Dictionary<string, Label>();

        bool _busy;
        DiffusionInstaller _boundInstaller;
        readonly ConcurrentQueue<string> _logQ = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<string> _serverQ = new ConcurrentQueue<string>();
        readonly List<string> _logLines = new List<string>();
        readonly List<string> _serverLines = new List<string>();
        volatile float _installFraction;
        volatile string _installMessage = "";

        // presets page state
        DropdownField _presetDrop;
        TextField _presetName;
        Label _presetStatus;

        static string Arg(string name)
        {
            var a = Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length; i++)
                if (a[i] == name) return i + 1 < a.Length && !a[i + 1].StartsWith("-") ? a[i + 1] : "";
            return null;
        }

        // =============================================================== build
        void Start()
        {
            _c = GetComponent<DiffusionController>();

            var settings = Resources.Load<PanelSettings>("StreamDiffusionPanelSettings");
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<PanelSettings>();
                settings.scaleMode = PanelScaleMode.ConstantPixelSize;
                settings.scale = 1f;
            }
            _doc = gameObject.AddComponent<UIDocument>();
            _doc.panelSettings = settings;
            var root = _doc.rootVisualElement;
            root.style.flexGrow = 1;

            AddSheet(root, "UI/Styles/DesignSystem/DesignSystem");
            AddSheet(root, "UI/Styles/SdGreyscale");
            AddSheet(root, "DsFonts/NotoSans/NotoSans");
            root.AddToClassList("ds-root");

            _hideTarget = new VisualElement();
            _hideTarget.style.flexGrow = 1;
            root.Add(_hideTarget);

            if (showHeader) BuildHeader(_hideTarget);

            _console = new DiffusionConsole(_c, openConsoleAtStart);
            _hideTarget.Add(_console.Strip);   // activity / progress strip under the header

            var main = new VisualElement();
            main.AddToClassList("sd-page");
            main.style.flexDirection = FlexDirection.Row;
            main.style.flexGrow = 1;
            main.style.paddingLeft = 16; main.style.paddingRight = 16; main.style.paddingTop = 14; main.style.paddingBottom = 14;
            _hideTarget.Add(main);

            var left = new VisualElement();
            left.style.flexGrow = 1;
            left.style.flexShrink = 1;
            left.style.minWidth = 380;
            main.Add(left);
            BuildPages(left);

            if (showOutput) { _outputCard = BuildOutput(); main.Add(_outputCard); }
            if (showConsole) _hideTarget.Add(_console.Dock);

            var toggle = new Button(() =>
            {
                bool hide = _hideTarget.style.display != DisplayStyle.None;
                _hideTarget.style.display = hide ? DisplayStyle.None : DisplayStyle.Flex;
                // the root paints the opaque page background: make it transparent while the panel is hidden
                // so the scene (main camera) shows through
                root.EnableInClassList("ds-root--hud", hide);
            });
            toggle.AddToClassList("ds-btn"); toggle.AddToClassList("ds-btn--icon");
            var cog = new VisualElement();
            cog.AddToClassList("ds-icon"); cog.AddToClassList("ds-icon--settings"); cog.AddToClassList("ds-icon--md");
            cog.pickingMode = PickingMode.Ignore;
            toggle.Add(cog);
            toggle.style.position = Position.Absolute; toggle.style.right = 12; toggle.style.top = 10;
            toggle.tooltip = "StreamDiffusionUnity Settings";
            if (showHideButton) root.Add(toggle);
            if (startHidden) { _hideTarget.style.display = DisplayStyle.None; root.AddToClassList("ds-root--hud"); }

            DesignSystemBehaviourBase<UIDocument>.EnsureToggleKnobs(root);

            var pageArg = Arg("-sd-page");
            if (!string.IsNullOrEmpty(pageArg) && Enum.TryParse(pageArg, true, out DiffusionPanelPage forced)) startPage = forced;
            SelectPage(startPage);

            InstallHoverTooltips(root);

            _c.Log += OnControllerLog;
            _c.PhaseChanged += _ => Refresh();
            Refresh();
        }

        // UI Toolkit does not show the 'tooltip' property in runtime panels. One shared label that follows the
        // pointer and shows the nearest ancestor's 'tooltip' text, so every element (also the ones created later,
        // like the rows of the LoRA / ControlNet lists) gets a tooltip just by setting .tooltip.
        void InstallHoverTooltips(VisualElement root)
        {
            var tip = new Label();
            tip.pickingMode = PickingMode.Ignore;
            tip.style.position = Position.Absolute;
            tip.style.paddingLeft = 8; tip.style.paddingRight = 8; tip.style.paddingTop = 4; tip.style.paddingBottom = 4;
            tip.style.backgroundColor = (Color)new Color32(31, 31, 31, 255);
            tip.style.borderTopWidth = tip.style.borderBottomWidth = tip.style.borderLeftWidth = tip.style.borderRightWidth = 1;
            tip.style.borderTopColor = tip.style.borderBottomColor = tip.style.borderLeftColor = tip.style.borderRightColor = (Color)new Color32(58, 58, 58, 255);
            tip.style.borderTopLeftRadius = tip.style.borderTopRightRadius = tip.style.borderBottomLeftRadius = tip.style.borderBottomRightRadius = 6;
            tip.style.fontSize = 12;
            tip.style.maxWidth = 460;
            tip.style.whiteSpace = WhiteSpace.Normal;
            tip.style.display = DisplayStyle.None;
            root.Add(tip);

            string current = null;
            root.RegisterCallback<PointerMoveEvent>(e =>
            {
                string text = null;
                for (var v = e.target as VisualElement; v != null && v != root; v = v.parent)
                    if (!string.IsNullOrEmpty(v.tooltip)) { text = v.tooltip; break; }
                if (string.IsNullOrEmpty(text))
                {
                    if (current != null) { current = null; tip.style.display = DisplayStyle.None; }
                    return;
                }
                if (text != current) { current = text; tip.text = text; }
                var p = root.WorldToLocal(e.position);
                float w = float.IsNaN(tip.resolvedStyle.width) ? 0f : tip.resolvedStyle.width;
                float x = Mathf.Min(p.x + 14f, Mathf.Max(8f, root.layout.width - w - 8f));
                tip.style.left = x;
                tip.style.top = p.y + 20f;
                tip.style.display = DisplayStyle.Flex;
                tip.BringToFront();
            }, TrickleDown.TrickleDown);
            root.RegisterCallback<PointerLeaveEvent>(_ => { current = null; tip.style.display = DisplayStyle.None; });
            root.RegisterCallback<PointerDownEvent>(_ => { current = null; tip.style.display = DisplayStyle.None; }, TrickleDown.TrickleDown);
        }

        void OnDestroy()
        {
            if (_c != null) _c.Log -= OnControllerLog;
            _console?.Dispose();
        }

        void OnControllerLog(string line)
        {
            if (line != null && line.StartsWith("[server]")) _serverQ.Enqueue(line);
        }

        static void AddSheet(VisualElement root, string resource)
        {
            var s = Resources.Load<StyleSheet>(resource);
            if (s != null) root.styleSheets.Add(s);
            else Debug.LogWarning("[SD] style sheet not found: " + resource);
        }

        void BuildHeader(VisualElement parent)
        {
            var header = new VisualElement();
            header.AddToClassList("sd-header");
            header.style.borderBottomWidth = 1;
            header.style.borderBottomColor = (Color)new Color32(42, 42, 42, 255);
            header.style.paddingTop = 16; header.style.paddingBottom = 16;
            header.style.paddingLeft = 24; header.style.paddingRight = 80;
            header.style.flexDirection = FlexDirection.Row;
            header.style.alignItems = Align.Center;

            var col = new VisualElement();
            col.style.flexGrow = 1;
            VisualElement title;
            if (string.IsNullOrWhiteSpace(titleText) || titleText.Trim() == "StreamDiffusionUnity")
                title = new DiffusionWordmark(54f, Color.white);          // the logo, instead of the plain text
            else
            {
                var custom = new VisualElement();
                var label = new Label(titleText);
                label.AddToClassList("ds-h1");
                label.style.fontSize = 44;
                var sub = new Label("Interactive Realtime Ai Generation");
                sub.AddToClassList("ds-body-2");
                sub.style.fontSize = 18;
                sub.style.color = Muted;
                custom.Add(label); custom.Add(sub);
                title = custom;
            }
            col.Add(title);
            header.Add(col);

            parent.Add(header);
        }

        // =============================================================== pages + tabs
        // ---- engine build confirmation (visible from every page) ----
        VisualElement _buildCard;
        Label _buildText, _buildPathLabel;

        VisualElement BuildBuildCard()
        {
            var card = Section("ENGINE BUILD NEEDED");
            card.style.display = DisplayStyle.None;
            card.style.borderLeftWidth = 3;
            card.style.borderLeftColor = (Color)new Color32(245, 158, 11, 255);

            _buildText = new Label("");
            _buildText.AddToClassList("ds-body-2");
            _buildText.style.whiteSpace = WhiteSpace.Normal;
            _buildText.style.marginBottom = 6;
            card.Add(_buildText);
            card.Add(Caption("Building keeps the GPU at full load for several minutes. Building while StreamDiffusionUnity is open could " +
                             "reset the GPU driver (black screen). The safer way is to create the build script and run it " +
                             "with StreamDiffusionUnity closed; afterwards this setup loads instantly."));

            var row = Row(card);
            var script = Btn("Create build script (run with Unity closed)", "primary", OnCreateBuildScript);
            script.style.marginRight = 6;
            var build = Btn("Build now", "secondary", () => _c.ConfirmEngineBuild());
            build.style.marginRight = 6;
            row.Add(script); row.Add(build);
            row.Add(Btn("Cancel", "ghost", () => _c.DeclineEngineBuild()));

            _buildPathLabel = Caption("");
            _buildPathLabel.style.marginTop = 6;
            card.Add(_buildPathLabel);

            var tdr = Caption("The GPU has reset twice during engine builds on this machine. Windows is using its default " +
                              "2 second GPU timeout (TdrDelay), which long TensorRT kernels can exceed while a 3D app is open. " +
                              "Raise it to 60 s once (admin command prompt, then reboot):" + System.Environment.NewLine +
                              @"reg add HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers /v TdrDelay /t REG_DWORD /d 60 /f" + System.Environment.NewLine +
                              @"reg add HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers /v TdrDdiDelay /t REG_DWORD /d 60 /f");
            tdr.style.color = (Color)new Color32(245, 180, 60, 255);
            tdr.style.marginTop = 6;
            tdr.style.display = DisplayStyle.None;
            card.Add(tdr);
            var tdrBtn = Btn("Fix GPU timeout (asks for administrator permission)", "secondary", OnFixGpuTimeout);
            tdrBtn.style.marginTop = 6;
            tdrBtn.style.display = DisplayStyle.None;
            card.Add(tdrBtn);
            _binders.Add(() =>
            {
                bool show = _c.PendingBuild != null && _c.GpuTimeout == GpuTimeoutState.Short;
                tdr.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
                tdrBtn.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
            });

            _binders.Add(() =>
            {
                var pb = _c.PendingBuild;
                _buildCard.style.display = pb != null ? DisplayStyle.Flex : DisplayStyle.None;
                if (pb != null)
                    _buildText.text = $"{pb.width} x {pb.height}, {pb.steps} step(s) has no cached engines: {pb.MissingText}. " +
                                      $"Roughly {pb.estimateMinutes:0} minutes of heavy GPU use.";
            });
            return card;
        }

        void OnCreateBuildScript()
        {
            try
            {
                string bat = _c.CreateEngineBuildScript();
                _buildPathLabel.text = "Script written: " + bat + "  (close Unity, then double-click it)";
                System.Diagnostics.Process.Start("explorer.exe", "/select,\"" + bat + "\"");
            }
            catch (Exception ex) { _buildPathLabel.text = "Could not create the script: " + ex.Message; }
        }

        void BuildPages(VisualElement parent)
        {
            _buildCard = BuildBuildCard();
            _buildCard.style.marginBottom = 12;
            parent.Add(_buildCard);

            var tabs = new VisualElement();
            tabs.AddToClassList("ds-tabs");
            tabs.style.marginBottom = 12;
            parent.Add(tabs);
            _tabsBar = tabs;

            var body = new ScrollView(ScrollViewMode.Vertical);
            body.style.flexGrow = 1;
            parent.Add(body);
            _pageScroll = body;
            // after layout: make the tab bar as wide as the cards below it and level with the OUTPUT card
            tabs.RegisterCallback<GeometryChangedEvent>(_ => AlignTabs());
            body.RegisterCallback<GeometryChangedEvent>(_ => AlignTabs());

            void Add(bool enabled, DiffusionPanelPage page, string title, Func<VisualElement> build)
            {
                if (!enabled) return;
                var tab = new Button { text = title };
                tab.AddToClassList("ds-tab");
                tab.clicked += () => SelectPage(page);
                tabs.Add(tab);
                var pageBody = build();
                pageBody.style.display = DisplayStyle.None;
                body.Add(pageBody);
                _pages.Add((page, tab, pageBody));
            }

            Add(showStream, DiffusionPanelPage.Stream, "Stream", BuildStreamPage);
            Add(showPerformance, DiffusionPanelPage.Performance, "Performance", BuildPerformancePage);
            Add(showModel, DiffusionPanelPage.Model, "Model", BuildModelPage);
            Add(showFx, DiffusionPanelPage.Fx, "FX", BuildFxPage);
            Add(showNdi, DiffusionPanelPage.Ndi, "NDI", BuildNdiPage);
            Add(showPresets, DiffusionPanelPage.Presets, "Presets", BuildPresetsPage);
            Add(showEnvironment, DiffusionPanelPage.Environment, "Environment", BuildEnvironmentPage);
            Add(showAbout, DiffusionPanelPage.About, "About", BuildAboutPage);

            if (_pages.Count <= 1) tabs.style.display = DisplayStyle.None;
        }

        VisualElement _tabsBar, _outputCard;
        ScrollView _pageScroll;
        bool _aligning;

        void AlignTabs()
        {
            if (_aligning || _tabsBar == null || _pageScroll == null) return;
            var t = _tabsBar.worldBound;
            if (t.width <= 0) return;
            _aligning = true;
            try
            {
                // sides: match the first visible card in the page
                VisualElement card = null;
                foreach (var (_, _, b) in _pages)
                    if (b.style.display != DisplayStyle.None && b.childCount > 0) { card = b[0]; break; }
                if (card != null && card.worldBound.width > 0)
                {
                    var c = card.worldBound;
                    float dl = c.x - t.x, dr = t.xMax - c.xMax;
                    if (Mathf.Abs(dl) > 0.5f) _tabsBar.style.marginLeft = _tabsBar.resolvedStyle.marginLeft + dl;
                    if (Mathf.Abs(dr) > 0.5f) _tabsBar.style.marginRight = _tabsBar.resolvedStyle.marginRight + dr;
                }
                // top: level with the OUTPUT card (only when the build card is not pushing the tabs down)
                if (_outputCard != null && _outputCard.worldBound.width > 0 && _buildCard.style.display == DisplayStyle.None)
                {
                    float dy = _outputCard.worldBound.y - t.y;
                    if (Mathf.Abs(dy) > 0.5f) _tabsBar.style.marginTop = _tabsBar.resolvedStyle.marginTop + dy;
                }
            }
            finally { _aligning = false; }
        }

        void SelectPage(DiffusionPanelPage page)
        {
            bool found = false;
            foreach (var p in _pages) if (p.page == page) found = true;
            if (!found && _pages.Count > 0) page = _pages[0].page;
            foreach (var (pg, tab, body) in _pages)
            {
                bool on = pg == page;
                body.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
                if (on) tab.AddToClassList("is-active"); else tab.RemoveFromClassList("is-active");
            }
        }

        // =============================================================== small builders
        static VisualElement Page()
        {
            var v = new VisualElement();
            v.style.flexDirection = FlexDirection.Column;
            return v;
        }

        static VisualElement Section(string title)
        {
            var s = new VisualElement();
            s.AddToClassList("ds-section");
            s.style.marginBottom = 12;
            var t = new Label(title);
            t.AddToClassList("ds-section__title");
            s.Add(t);
            return s;
        }

        static VisualElement Row(VisualElement parent = null)
        {
            var r = new VisualElement();
            r.style.flexDirection = FlexDirection.Row;
            r.style.alignItems = Align.Center;
            parent?.Add(r);
            return r;
        }

        static Label Caption(string text)
        {
            var l = new Label(text);
            l.AddToClassList("ds-caption");
            l.style.color = Muted;
            l.style.whiteSpace = WhiteSpace.Normal;
            l.style.marginBottom = 6;
            return l;
        }

        static VisualElement InfoRow(string label, out Label value)
        {
            var r = new VisualElement();
            r.AddToClassList("ds-info-row");
            var left = new VisualElement();
            left.AddToClassList("ds-info-row__left");
            var l = new Label(label);
            l.AddToClassList("ds-info-row__label");
            left.Add(l);
            r.Add(left);
            value = new Label("-");
            value.AddToClassList("ds-info-row__value");
            r.Add(value);
            return r;
        }

        static Button Btn(string text, string kind, Action click)
        {
            var b = new Button(click) { text = text };
            b.AddToClassList("ds-btn");
            b.AddToClassList("ds-btn--" + kind);
            return b;
        }

        VisualElement ToggleRow(string label, Func<bool> get, Action<bool> set)
        {
            var r = new VisualElement();
            r.AddToClassList("ds-info-row");
            var left = new VisualElement();
            left.AddToClassList("ds-info-row__left");
            var l = new Label(label);
            l.AddToClassList("ds-info-row__label");
            left.Add(l);
            r.Add(left);
            var t = new Toggle { value = get() };
            t.AddToClassList("ds-toggle");
            t.RegisterValueChangedCallback(e => set(e.newValue));
            r.Add(t);
            _binders.Add(() => { if (t.value != get()) t.SetValueWithoutNotify(get()); });
            return r;
        }

        VisualElement SliderRow(string label, float min, float max, Func<float> get, Action<float> set,
            string fmt = "0.00", float marginBottom = 10f)
        {
            var box = new VisualElement();
            box.style.marginBottom = marginBottom;
            var head = Row(box);
            var l = new Label(label);
            l.AddToClassList("ds-body-2");
            l.style.flexGrow = 1;
            var v = new Label(get().ToString(fmt));
            v.AddToClassList("ds-body-2");
            v.style.color = Muted;
            head.Add(l); head.Add(v);
            var s = new Slider(min, max) { value = get() };
            s.AddToClassList("ds-slider"); s.AddToClassList("ds-slider--filled");
            s.RegisterValueChangedCallback(e => { v.text = e.newValue.ToString(fmt); set(e.newValue); });
            box.Add(s);
            _binders.Add(() =>
            {
                float cur = get();
                if (!Mathf.Approximately(s.value, cur)) { s.SetValueWithoutNotify(cur); v.text = cur.ToString(fmt); }
            });
            return box;
        }

        TextField Input(Func<string> get, Action<string> set, bool multiline = false, float minHeight = 70f)
        {
            var f = new TextField { value = get(), multiline = multiline };
            f.AddToClassList("ds-input");
            if (multiline)
            {
                f.AddToClassList("ds-textarea");
                f.style.minHeight = minHeight;
                // the design system sets min-height: 96px on the inner input element itself
                var inner = f.Q(className: "unity-text-input") ?? f.Q(className: "unity-base-text-field__input");
                if (inner != null) inner.style.minHeight = minHeight;
            }
            bool focused = false;
            f.RegisterCallback<FocusInEvent>(_ => focused = true);
            f.RegisterCallback<FocusOutEvent>(_ => focused = false);
            f.RegisterValueChangedCallback(e => set(e.newValue));
            _binders.Add(() => { if (!focused && f.value != get()) f.SetValueWithoutNotify(get()); });
            return f;
        }

        IntegerField IntInput(Func<int> get, Action<int> set)
        {
            var f = new IntegerField { value = get() };
            f.AddToClassList("ds-input");
            bool focused = false;
            f.RegisterCallback<FocusInEvent>(_ => focused = true);
            f.RegisterCallback<FocusOutEvent>(_ => focused = false);
            f.RegisterValueChangedCallback(e => set(e.newValue));
            _binders.Add(() => { if (!focused && f.value != get()) f.SetValueWithoutNotify(get()); });
            return f;
        }

        FloatField FloatInput(Func<float> get, Action<float> set)
        {
            var f = new FloatField { value = get() };
            f.AddToClassList("ds-input");
            bool focused = false;
            f.RegisterCallback<FocusInEvent>(_ => focused = true);
            f.RegisterCallback<FocusOutEvent>(_ => focused = false);
            f.RegisterValueChangedCallback(e => set(e.newValue));
            _binders.Add(() => { if (!focused && !Mathf.Approximately(f.value, get())) f.SetValueWithoutNotify(get()); });
            return f;
        }

        DropdownField Dropdown(List<string> choices, Func<int> getIndex, Action<int> setIndex)
        {
            int idx = Mathf.Clamp(getIndex(), 0, choices.Count - 1);
            var d = new DropdownField(choices, idx);
            d.AddToClassList("ds-dropdown");
            d.RegisterValueChangedCallback(_ => setIndex(d.index));
            _binders.Add(() =>
            {
                int cur = Mathf.Clamp(getIndex(), 0, choices.Count - 1);
                if (d.index != cur) d.SetValueWithoutNotify(choices[cur]);
            });
            return d;
        }

        static readonly List<string> InterpNames = new List<string> { "Slerp", "Average", "Cosine weighted" };

        // =============================================================== STREAM page
        VisualElement _promptRows, _seedRows, _promptOpts, _seedOpts;
        int _builtPromptRows = -1, _builtSeedRows = -1;

        // step schedule
        VisualElement _stepRows, _stepHint;
        Label _stepHintLabel;
        int _builtSteps = -1;

        int MaxStepPosition() => Mathf.Max(2, _c.numInferenceSteps - 1);   // positions are 1 .. numInferenceSteps-1

        void RebuildStepRows()
        {
            _stepRows.Clear();
            if (_c.tIndexList == null || _c.tIndexList.Length == 0) _c.tIndexList = new[] { 1 };
            _builtSteps = _c.tIndexList.Length;
            for (int i = 0; i < _c.tIndexList.Length; i++)
            {
                int idx = i;
                _stepRows.Add(SliderRow($"Step {i + 1}", 1, MaxStepPosition(),
                    () => idx < _c.tIndexList.Length ? _c.tIndexList[idx] : 1,
                    v => { if (idx < _c.tIndexList.Length) _c.tIndexList[idx] = Mathf.Clamp(Mathf.RoundToInt(v), 1, MaxStepPosition()); },
                    "0"));
            }
        }

        VisualElement BuildStreamPage()
        {
            var page = Page();

            // ---- prompts: first row is the main prompt, two or more rows blend ----
            var prompt = Section("PROMPTS");
            prompt.Add(Caption("The first row is the main prompt. Add rows to blend several prompts by weight."));
            _promptRows = new VisualElement();
            prompt.Add(_promptRows);
            RebuildPromptRows();

            var pbar = Row(prompt);
            pbar.style.marginTop = 2; pbar.style.marginBottom = 8;
            var padd = Btn("Add prompt", "secondary", () => { _c.promptBlend.Add(new PromptBlendEntry("", 0.5f)); });
            padd.style.marginRight = 6;
            pbar.Add(padd);
            pbar.Add(Btn("Remove last", "ghost", () =>
            {
                if (_c.promptBlend.Count > 1) _c.promptBlend.RemoveAt(_c.promptBlend.Count - 1);
            }));

            _promptOpts = new VisualElement();
            var pi = Dropdown(InterpNames, () => (int)_c.promptInterpolation, i => _c.promptInterpolation = (BlendInterpolation)i);
            pi.style.marginBottom = 6;
            _promptOpts.Add(pi);
            _promptOpts.Add(ToggleRow("Normalize weights", () => _c.normalizePromptWeights, v => _c.normalizePromptWeights = v));
            prompt.Add(_promptOpts);

            var neg = Caption("Negative prompt");
            neg.style.marginTop = 8;
            prompt.Add(neg);
            prompt.Add(Input(() => _c.negativePrompt, v => _c.negativePrompt = v, multiline: true, minHeight: 42f));
            page.Add(prompt);

            // ---- step schedule (TouchDesigner: "Step Schedule") ----
            var steps = Section("STEP SCHEDULE");
            steps.Add(Caption("Works like 'denoise' in img2img. Each step has a position: lower positions start denoising " +
                              "sooner and add more randomness (more AI), higher positions keep the result close to the input. " +
                              "You can move the steps while streaming."));
            _stepRows = new VisualElement();
            steps.Add(_stepRows);
            RebuildStepRows();

            var sbtn = Row(steps);
            sbtn.style.marginTop = 2; sbtn.style.marginBottom = 8;
            var plus = Btn("+ Add step", "secondary", () =>
            {
                var cur = _c.tIndexList != null && _c.tIndexList.Length > 0 ? _c.tIndexList : new[] { 1 };
                var next = new int[Mathf.Min(cur.Length + 1, 8)];
                Array.Copy(cur, next, Mathf.Min(cur.Length, next.Length));
                if (next.Length > cur.Length)
                    next[next.Length - 1] = Mathf.Clamp(cur[cur.Length - 1] + 12, 1, MaxStepPosition());
                _c.tIndexList = next;
            });
            plus.style.marginRight = 6;
            sbtn.Add(plus);
            sbtn.Add(Btn("- Remove step", "ghost", () =>
            {
                var cur = _c.tIndexList;
                if (cur != null && cur.Length > 1)
                {
                    var next = new int[cur.Length - 1];
                    Array.Copy(cur, next, next.Length);
                    _c.tIndexList = next;
                }
            }));

            _stepHint = new VisualElement();
            _stepHint.style.display = DisplayStyle.None;
            _stepHintLabel = Caption("");
            _stepHint.Add(_stepHintLabel);
            var apply = Btn("Apply step count (reload model)", "primary", OnReload);
            apply.AddToClassList("ds-btn--block");
            _stepHint.Add(apply);
            steps.Add(_stepHint);
            page.Add(steps);

            _binders.Add(() =>
            {
                if (_c.tIndexList == null || _c.tIndexList.Length == 0) _c.tIndexList = new[] { 1 };
                if (_builtSteps != _c.tIndexList.Length) RebuildStepRows();
                bool pending = _c.StepCountPending;
                _stepHint.style.display = pending ? DisplayStyle.Flex : DisplayStyle.None;
                if (pending)
                    _stepHintLabel.text = $"Step count changed ({_c.LoadedStepCount} -> {_c.tIndexList.Length}). " +
                                          "The model's engine is built for a fixed number of steps: reload to apply " +
                                          "(a new step count builds its engine once, then it is cached).";
            });

            // ---- seeds: first row is the main seed, two or more rows blend ----
            var seed = Section("SEED");
            seed.Add(Caption("The first row is the main seed. Add rows to blend the noise of several seeds."));
            _seedRows = new VisualElement();
            seed.Add(_seedRows);
            RebuildSeedRows();

            var sbar = Row(seed);
            sbar.style.marginTop = 2; sbar.style.marginBottom = 8;
            var rnd = Btn("Randomize first", "secondary", () => _c.seed = UnityEngine.Random.Range(0, int.MaxValue));
            rnd.style.marginRight = 6;
            sbar.Add(rnd);
            var sadd = Btn("Add seed", "secondary", () => { _c.seedBlend.Add(new SeedBlendEntry(UnityEngine.Random.Range(0, 100000), 0.5f)); });
            sadd.style.marginRight = 6;
            sbar.Add(sadd);
            sbar.Add(Btn("Remove last", "ghost", () =>
            {
                if (_c.seedBlend.Count > 1) _c.seedBlend.RemoveAt(_c.seedBlend.Count - 1);
            }));

            _seedOpts = new VisualElement();
            var si = Dropdown(InterpNames, () => (int)_c.seedInterpolation, i => _c.seedInterpolation = (BlendInterpolation)i);
            si.style.marginBottom = 6;
            _seedOpts.Add(si);
            _seedOpts.Add(ToggleRow("Normalize weights", () => _c.normalizeSeedWeights, v => _c.normalizeSeedWeights = v));
            seed.Add(_seedOpts);
            page.Add(seed);

            // ---- IP-Adapter live controls ----
            var ipCard = Section("IP-ADAPTER");
            ipCard.Add(Caption("Image prompt. Scale and the on/off switch are live; the reference image comes from " +
                               "'Reference Texture' on the controller."));
            ipCard.Add(ToggleRow("Reference image active", () => _c.ipAdapterEnabled, v => _c.ipAdapterEnabled = v));
            ipCard.Add(SliderRow("Scale", 0f, 2f, () => _c.ipAdapterScale, v => _c.ipAdapterScale = v, "0.00"));
            ipCard.Add(SliderRow("Refresh reference (s, 0 = manual)", 0f, 10f, () => _c.referenceRefreshSeconds,
                v => _c.referenceRefreshSeconds = v, "0.0"));
            ipCard.Add(Btn("Update reference now", "secondary", () => _c.UpdateReferenceNow()));
            var ipSrc = Caption("");
            ipSrc.style.marginTop = 6;
            ipCard.Add(ipSrc);
            _binders.Add(() =>
            {
                ipCard.style.display = _c.useIpAdapter ? DisplayStyle.Flex : DisplayStyle.None;
                ipSrc.text = _c.referenceTexture == null
                    ? "No reference texture assigned: set 'Reference Texture' on the DiffusionController."
                    : "Reference: " + _c.referenceTexture.name;
            });
            page.Add(ipCard);

            // ---- ControlNet live controls; the set is managed on the Model page ----
            var cnCard = Section("CONTROLNET");
            cnCard.Add(Caption("Structure guidance made from the input picture. Strength and the on/off switch are live; " +
                               "add or remove ControlNets on the Model page."));
            cnCard.Add(ToggleRow("Show the control map (what the ControlNet sees)", () => _c.showControlMap,
                v => _c.showControlMap = v));
            cnCard.Add(Caption("Control image"));
            cnCard.Add(EnumDropdown(new[] { "Same as the input picture", "A separate texture (controlTexture)" },
                () => (int)_c.controlSource, i => _c.controlSource = (ControlSourceMode)i));
            var cnSrcNote = Caption("");
            cnCard.Add(cnSrcNote);
            _binders.Add(() =>
            {
                cnSrcNote.text = _c.controlSource != ControlSourceMode.Texture ? ""
                    : _c.controlTexture == null
                        ? "No texture assigned: set 'Control Texture' on the DiffusionController (e.g. a depth-map RenderTexture). " +
                          "Until then the input picture is used."
                        : "Using: " + _c.controlTexture.name + " (use the 'passthrough' preprocessor if it already is a depth, " +
                          "edge or pose map).";
            });
            _cnLiveRows = new VisualElement();
            cnCard.Add(_cnLiveRows);
            _binders.Add(() =>
            {
                int n = _c.controlNets != null ? _c.controlNets.Count : 0;
                cnCard.style.display = n > 0 ? DisplayStyle.Flex : DisplayStyle.None;
                string key = CnRowsKey();
                if (key != _cnLiveBuilt) { _cnLiveBuilt = key; RebuildCnLiveRows(); }
            });
            page.Add(cnCard);

            // ---- LoRA weights (live); the set is managed on the Model page ----
            var loraCard = Section("LORA");
            loraCard.Add(Caption("Style adapters. The weight changes live; add or remove LoRAs on the Model page."));
            _loraWeightRows = new VisualElement();
            loraCard.Add(_loraWeightRows);
            _binders.Add(() =>
            {
                int n = _c.loras != null ? _c.loras.Count : 0;
                loraCard.style.display = n > 0 ? DisplayStyle.Flex : DisplayStyle.None;
                string key = LoraRowsKey();
                if (key != _loraWeightsBuilt) { _loraWeightsBuilt = key; RebuildLoraWeightRows(); }
            });
            page.Add(loraCard);

            // ---- temporal consistency (StreamV2V): live tuning, shown when the option is on ----
            var tmp = Section("TEMPORAL CONSISTENCY (STREAMV2V)");
            tmp.Add(Caption("Live tuning of the cross-frame memory (switch the feature on in the Model page; it needs its own " +
                            "engine). The effect is subtle, and strongest with several denoising steps: a single-step " +
                            "model is already stable. Keep 'Memory frames used' at 2 or more."));
            tmp.Add(SliderRow("Strength", 0f, 1f, () => _c.fiStrength, v => _c.fiStrength = v, "0.00"));
            tmp.Add(SliderRow("Similarity gate", 0f, 1f, () => _c.fiThreshold, v => _c.fiThreshold = v, "0.00"));
            tmp.Add(SliderRow("Memory interval (frames)", 1, 10, () => _c.v2vCacheInterval,
                v => _c.v2vCacheInterval = Mathf.RoundToInt(v), "0"));
            tmp.Add(SliderRow("Memory frames used", 2, 8, () => _c.v2vCacheFrames,
                v => _c.v2vCacheFrames = Mathf.RoundToInt(v), "0"));
            _binders.Add(() => tmp.style.display = _c.useStreamV2V ? DisplayStyle.Flex : DisplayStyle.None);
            page.Add(tmp);

            // rows follow the inspector: rebuild when the list size changes, hide blend options for a single entry
            _binders.Add(() =>
            {
                if (_c.promptBlend == null || _c.seedBlend == null) return;
                if (_builtPromptRows != _c.promptBlend.Count) RebuildPromptRows();
                if (_builtSeedRows != _c.seedBlend.Count) RebuildSeedRows();
                _promptOpts.style.display = _c.promptBlend.Count >= 2 ? DisplayStyle.Flex : DisplayStyle.None;
                _seedOpts.style.display = _c.seedBlend.Count >= 2 ? DisplayStyle.Flex : DisplayStyle.None;
            });
            return page;
        }

        void RebuildPromptRows()
        {
            _promptRows.Clear();
            if (_c.promptBlend == null) _c.promptBlend = new List<PromptBlendEntry>();
            if (_c.promptBlend.Count == 0) _c.promptBlend.Add(new PromptBlendEntry("", 1f));
            _builtPromptRows = _c.promptBlend.Count;
            bool multi = _c.promptBlend.Count >= 2;
            for (int i = 0; i < _c.promptBlend.Count; i++)
            {
                int idx = i;
                var item = new VisualElement();
                item.style.marginBottom = multi ? 8 : 6;
                _promptRows.Add(item);

                // single prompt keeps a roomy box; blend rows are compact (about two lines)
                var t = Input(() => idx < _c.promptBlend.Count ? _c.promptBlend[idx].text : "", v =>
                {
                    if (idx < _c.promptBlend.Count) _c.promptBlend[idx].text = v;
                }, multiline: true, minHeight: multi ? 42f : 70f);
                t.style.marginBottom = multi ? 2 : 0;
                item.Add(t);

                if (multi)
                    item.Add(SliderRow("Weight", 0f, 1f,
                        () => idx < _c.promptBlend.Count ? _c.promptBlend[idx].weight : 0f,
                        v => { if (idx < _c.promptBlend.Count) _c.promptBlend[idx].weight = v; },
                        "0.00", 0f));
            }
        }

        void RebuildSeedRows()
        {
            _seedRows.Clear();
            if (_c.seedBlend == null) _c.seedBlend = new List<SeedBlendEntry>();
            if (_c.seedBlend.Count == 0) _c.seedBlend.Add(new SeedBlendEntry(0, 1f));
            _builtSeedRows = _c.seedBlend.Count;
            bool multi = _c.seedBlend.Count >= 2;
            for (int i = 0; i < _c.seedBlend.Count; i++)
            {
                int idx = i;
                var item = new VisualElement();
                item.style.marginBottom = multi ? 8 : 6;
                _seedRows.Add(item);

                var sf = IntInput(() => idx < _c.seedBlend.Count ? _c.seedBlend[idx].seed : 0, v =>
                {
                    if (idx < _c.seedBlend.Count) _c.seedBlend[idx].seed = v;
                });
                sf.style.marginBottom = multi ? 2 : 0;
                item.Add(sf);

                if (multi)
                    item.Add(SliderRow("Weight", 0f, 1f,
                        () => idx < _c.seedBlend.Count ? _c.seedBlend[idx].weight : 0f,
                        v => { if (idx < _c.seedBlend.Count) _c.seedBlend[idx].weight = v; },
                        "0.00", 0f));
            }
        }

        // =============================================================== PERFORMANCE page
        VisualElement BuildPerformancePage()
        {
            var page = Page();

            var sim = Section("SIMILAR FRAME FILTER");
            sim.Add(ToggleRow("Skip near-identical frames", () => _c.similarFilterEnabled, v => _c.similarFilterEnabled = v));
            sim.Add(Caption("Saves GPU when the source barely changes. The previous result is repeated for skipped frames."));
            sim.Add(SliderRow("Similarity threshold", 0.5f, 1f, () => _c.similarFilterThreshold, v => _c.similarFilterThreshold = v, "0.000"));
            sim.Add(SliderRow("Max consecutive skipped frames", 1, 60, () => _c.similarFilterMaxSkip,
                v => _c.similarFilterMaxSkip = Mathf.RoundToInt(v), "0"));
            page.Add(sim);

            var smp = Section("SAMPLING AND RATE");
            smp.Add(ToggleRow("Add noise", () => _c.addNoise, v => _c.addNoise = v));
            smp.Add(Caption("Leave on unless you know you need it off (it degrades multi-step output)."));
            smp.Add(ToggleRow("Low-latency readback (synchronous)", () => _c.syncReadback, v => _c.syncReadback = v));
            smp.Add(Caption("Reads the picture from the GPU without the one-to-two frame delay of the asynchronous " +
                            "readback: about 12 ms less latency, at the cost of a short wait on the main thread. " +
                            "Turn it off if your scene is GPU-heavy and Unity's own frame rate drops."));
            smp.Add(SliderRow("Max send FPS (0 = every frame)", 0, 120, () => _c.maxSendFps, v => _c.maxSendFps = Mathf.RoundToInt(v), "0"));
            page.Add(smp);
            return page;
        }

        // =============================================================== MODEL page
        static readonly string[] ResPresets =
        {
            "512 x 512", "640 x 640", "768 x 768", "1024 x 1024",
            "768 x 512", "512 x 768", "1024 x 512", "512 x 1024", "1024 x 576",
        };
        static readonly int[][] ResValues =
        {
            new[] { 512, 512 }, new[] { 640, 640 }, new[] { 768, 768 }, new[] { 1024, 1024 },
            new[] { 768, 512 }, new[] { 512, 768 }, new[] { 1024, 512 }, new[] { 512, 1024 }, new[] { 1024, 576 },
        };
        const string CustomResolution = "Custom (use the width / height fields)";

        VisualElement BuildModelPage()
        {
            var page = Page();
            var m = Section("MODEL");
            m.Add(Caption("Model"));
            _modelDrop = new DropdownField(new List<string> { "(custom id below)" }, 0);
            _modelDrop.AddToClassList("ds-dropdown");
            _modelDrop.style.flexGrow = 1;
            _modelDrop.RegisterValueChangedCallback(_ => OnModelPicked());
            var modelRow = Row(m);
            modelRow.style.marginBottom = 4;
            modelRow.Add(_modelDrop);
            var mrefresh = Btn("Refresh", "ghost", () => RefreshModelList());
            mrefresh.style.marginLeft = 6; mrefresh.style.marginRight = 6;
            modelRow.Add(mrefresh);
            modelRow.Add(Btn("Open folder", "ghost", () =>
            {
                string problem = _c.OpenModelFolder();
                if (problem != null) _modelFolderLabel.text = problem;
            }));
            _modelFolderLabel = Caption("");
            m.Add(_modelFolderLabel);
            RefreshModelList();
            _binders.Add(SyncModelDropdown);
            m.Add(Caption("Model id (Hugging Face repo id, or a path to a local checkpoint)"));
            var id = Input(() => _c.modelId, v => _c.modelId = v);
            id.style.marginBottom = 8;
            m.Add(id);

            m.Add(Caption("Acceleration"));
            var acc = Dropdown(new List<string> { "tensorrt", "none" },
                () => _c.acceleration == "none" ? 1 : 0, i => _c.acceleration = i == 1 ? "none" : "tensorrt");
            acc.style.marginBottom = 8;
            m.Add(acc);
            m.Add(ToggleRow("Tiny VAE (faster, softer)", () => _c.useTinyVae, v => _c.useTinyVae = v));
            m.Add(ToggleRow("Temporal consistency (StreamV2V)", () => _c.useStreamV2V, v => _c.useStreamV2V = v));
            m.Add(Caption("Keeps the picture coherent from frame to frame. Builds its own engine (cached) and uses more " +
                          "VRAM; the resolution is then fixed. Tune it live on the Stream page."));
            m.Add(ToggleRow("Feature injection", () => _c.featureInjection, v => _c.featureInjection = v));

            m.Add(Caption("Resolution"));
            var presets = new List<string>(ResPresets) { CustomResolution };
            var resDrop = new DropdownField(presets, 0);
            resDrop.AddToClassList("ds-dropdown");
            resDrop.style.marginBottom = 6;
            resDrop.RegisterValueChangedCallback(_ =>
            {
                int i = resDrop.index;
                if (i >= 0 && i < ResValues.Length) { _c.width = ResValues[i][0]; _c.height = ResValues[i][1]; }
            });
            _binders.Add(() =>
            {
                // follow the width / height fields (inspector or typing): a preset when it matches, else Custom
                string cur = $"{_c.width} x {_c.height}";
                if (!presets.Contains(cur) || cur == CustomResolution) cur = CustomResolution;
                if (resDrop.value != cur) resDrop.SetValueWithoutNotify(cur);
            });
            m.Add(resDrop);
            var wh = Row(m);
            var wf = IntInput(() => _c.width, v => _c.width = v);
            wf.style.flexGrow = 1; wf.style.marginRight = 6;
            var hf = IntInput(() => _c.height, v => _c.height = v);
            hf.style.flexGrow = 1;
            wh.Add(wf); wh.Add(hf);
            page.Add(m);

            var cn = Section("CONTROLNET");
            cn.Add(Caption("Guides the picture with edges, depth or pose taken from the input. Each ControlNet must match the " +
                           "model family and downloads once (about 2.5 GB for SDXL). Adding or removing one needs 'Apply and " +
                           "reload' (engine build, cached); strength is live on the Stream page."));
            var cnPick = Row(cn);
            _cnDrop = new DropdownField(new List<string>(CnLabels), 0);
            _cnDrop.AddToClassList("ds-dropdown");
            _cnDrop.style.flexGrow = 1; _cnDrop.style.marginRight = 6;
            cnPick.Add(_cnDrop);
            cnPick.Add(Btn("Add", "secondary", OnAddControlNet));
            var cnCustom = Row(cn);
            cnCustom.style.marginTop = 6;
            _cnCustomId = new TextField { value = "" };
            _cnCustomId.AddToClassList("ds-input");
            _cnCustomId.style.flexGrow = 1; _cnCustomId.style.marginRight = 6;
            _cnCustomId.tooltip = "Any ControlNet repo id, e.g. xinsir/controlnet-canny-sdxl-1.0";
            cnCustom.Add(_cnCustomId);
            cnCustom.Add(Btn("Add custom id", "ghost", () =>
            {
                string id = (_cnCustomId.value ?? "").Trim();
                if (id.Length > 0) { _c.controlNets.Add(new ControlNetEntry(id, "canny", 0.5f)); _cnCustomId.value = ""; }
            }));
            _cnSetRows = new VisualElement();
            _cnSetRows.style.marginTop = 6;
            cn.Add(_cnSetRows);
            _cnSetNote = Caption("");
            cn.Add(_cnSetNote);
            _binders.Add(PollControlNets);
            page.Add(cn);

            var ip = Section("IP-ADAPTER (IMAGE PROMPT)");
            ip.Add(Caption("Steers the result with the look of a reference image (style, subject, colours) next to the " +
                           "text prompt. Needs its own engine (cached) and a model of the same family as the adapter. " +
                           "Assign the reference image to 'Reference Texture' on the DiffusionController."));
            ip.Add(ToggleRow("Use IP-Adapter", () => _c.useIpAdapter, v => _c.useIpAdapter = v));
            ip.Add(Caption("Adapter"));
            ip.Add(EnumDropdown(IpAdapterNames(),
                () => Mathf.Max(0, System.Array.FindIndex(IpAdapterChoices, c => c.model == (_c.ipAdapterModel ?? "").Trim())),
                i =>
                {
                    var c = IpAdapterChoices[Mathf.Clamp(i, 0, IpAdapterChoices.Length - 1)];
                    _c.ipAdapterModel = c.model; _c.ipAdapterEncoder = c.encoder;
                }));
            ip.Add(Caption("Plus follows the reference more closely (16 image tokens). FaceID keeps the identity of the face " +
                           "in the reference (needs a visible face; InsightFace is downloaded on first use)."));
            var ipNote = Caption("");
            ip.Add(ipNote);
            _binders.Add(() =>
            {
                bool changed = _c.LoadedIpAdapterKey != null && _c.LoadedIpAdapterKey != _c.IpAdapterKey();
                ipNote.text = changed ? "IP-Adapter setting changed: press 'Apply and reload model' below to use it." : "";
                ipNote.style.color = changed ? (Color)new Color32(245, 180, 60, 255) : Muted;
            });
            page.Add(ip);

            var adv = Section("ADVANCED");
            adv.Add(Caption("Engine options. They apply on 'Apply and reload model'; mode and CFG type change the engine " +
                            "(built once, then cached), scheduler and sampler do not."));
            adv.Add(Caption("Mode"));
            adv.Add(EnumDropdown(new[] { "Image to image (follows the input)", "Text to image (ignores the input)" },
                () => (int)_c.mode, i => _c.mode = (DiffusionMode)i));
            adv.Add(Caption("CFG type"));
            adv.Add(EnumDropdown(new[] { "Self (fast, default)", "Initialize", "Full (slower)", "None (no guidance)" },
                () => (int)_c.cfgType, i => _c.cfgType = (DiffusionCfgType)i));
            adv.Add(Caption("Scheduler"));
            adv.Add(EnumDropdown(new[] { "LCM (default)", "TCD" },
                () => (int)_c.scheduler, i => _c.scheduler = (DiffusionScheduler)i));
            adv.Add(Caption("Sampler"));
            adv.Add(EnumDropdown(new[] { "Normal (default)", "Simple", "SGM uniform", "DDIM", "Beta", "Karras" },
                () => (int)_c.sampler, i => _c.sampler = (DiffusionSampler)i));
            page.Add(adv);

            var lora = Section("LORA");
            lora.Add(Caption("Style adapters (.safetensors) found in the server's models/LoRA folder. They must match the " +
                             "model family. Adding or removing one needs 'Apply and reload' (engine rebuild, cached); the " +
                             "weights are live on the Stream page."));
            _loraFolderLabel = Caption("");
            lora.Add(_loraFolderLabel);
            var pick = Row(lora);
            _loraDrop = new DropdownField(new List<string> { "(connect to list)" }, 0);
            _loraDrop.AddToClassList("ds-dropdown");
            _loraDrop.style.flexGrow = 1; _loraDrop.style.marginRight = 6;
            pick.Add(_loraDrop);
            var addLora = Btn("Add", "secondary", OnAddLora);
            addLora.style.marginRight = 6;
            pick.Add(addLora);
            var refresh = Btn("Refresh", "ghost", () => RefreshLoraList());
            refresh.style.marginRight = 6;
            pick.Add(refresh);
            pick.Add(Btn("Open folder", "ghost", () =>
            {
                string problem = _c.OpenLoraFolder();
                if (problem != null) _loraFolderLabel.text = problem;
            }));
            _loraSetRows = new VisualElement();
            _loraSetRows.style.marginTop = 6;
            lora.Add(_loraSetRows);
            _loraSetNote = Caption("");
            lora.Add(_loraSetNote);
            _binders.Add(PollLoras);
            page.Add(lora);

            var apply = Section("APPLY");
            apply.Add(Caption("Model, acceleration, Tiny VAE and resolution only take effect when the model is reloaded. " +
                              "The first time a combination is used, TensorRT engines are built (minutes) and then cached."));
            apply.Add(InfoRow("Engines for this setup", out _engineState));
            _binders.Add(PollEngineStatus);
            var reload = Btn("Apply and reload model", "primary", OnReload);
            reload.AddToClassList("ds-btn--block");
            apply.Add(reload);
            apply.Add(Caption("Not available yet: preprocessor TensorRT engines."));
            page.Add(apply);
            return page;
        }

        // ---- LoRA UI ----
        // ---- ControlNet UI ----
        static readonly string[] CnIds =
        {
            "diffusers/controlnet-canny-sdxl-1.0", "diffusers/controlnet-depth-sdxl-1.0",
            "thibaud/controlnet-openpose-sdxl-1.0",
            "lllyasviel/control_v11p_sd15_canny", "lllyasviel/control_v11f1p_sd15_depth", "lllyasviel/control_v11p_sd15_openpose",
        };
        static readonly string[] CnPre = { "canny", "depth", "openpose", "canny", "depth", "openpose" };
        static readonly string[] CnLabels =
        {
            "SDXL - Canny edges", "SDXL - Depth (downloaded already)", "SDXL - OpenPose",
            "SD 1.5 - Canny edges", "SD 1.5 - Depth (downloaded already)", "SD 1.5 - OpenPose",
        };
        static readonly string[] CnPreprocessors =
            { "canny", "depth", "soft_edge", "standard_lineart", "lineart", "openpose", "mediapipe_pose", "passthrough" };
        DropdownField _cnDrop;
        TextField _cnCustomId;
        VisualElement _cnSetRows, _cnLiveRows;
        Label _cnSetNote;
        string _cnSetBuilt, _cnLiveBuilt;

        string CnRowsKey()
        {
            var sb = new System.Text.StringBuilder();
            if (_c.controlNets != null) foreach (var c in _c.controlNets) sb.Append(c?.modelId).Append(':').Append(c?.preprocessor).Append(';');
            return sb.ToString();
        }

        void OnAddControlNet()
        {
            int i = _cnDrop.index;
            if (i < 0 || i >= CnIds.Length) return;
            foreach (var c in _c.controlNets) if (c != null && c.modelId == CnIds[i]) return;
            _c.controlNets.Add(new ControlNetEntry(CnIds[i], CnPre[i], 0.5f));
        }

        void RebuildCnSetRows()
        {
            _cnSetRows.Clear();
            for (int i = 0; i < _c.controlNets.Count; i++)
            {
                int idx = i;
                var c = _c.controlNets[i];
                var item = new VisualElement();
                item.style.marginBottom = 6;
                var head = Row(item);
                var name = new Label(c.DisplayName);
                name.AddToClassList("ds-body-2");
                name.style.flexGrow = 1;
                name.tooltip = c.modelId;
                head.Add(name);
                head.Add(Btn("Remove", "ghost", () => { if (idx < _c.controlNets.Count) _c.controlNets.RemoveAt(idx); }));
                item.Add(Caption("Preprocessor (turns the input into the control image; applies on reload)"));
                var pre = new List<string>(CnPreprocessors);
                if (!pre.Contains(c.preprocessor)) pre.Insert(0, c.preprocessor);
                var d = new DropdownField(pre, Mathf.Max(0, pre.IndexOf(c.preprocessor)));
                d.AddToClassList("ds-dropdown");
                d.RegisterValueChangedCallback(_ => { if (idx < _c.controlNets.Count && d.index >= 0) _c.controlNets[idx].preprocessor = pre[d.index]; });
                item.Add(d);
                _cnSetRows.Add(item);
            }
        }

        void RebuildCnLiveRows()
        {
            _cnLiveRows.Clear();
            for (int i = 0; i < _c.controlNets.Count; i++)
            {
                int idx = i;
                var c = _c.controlNets[i];
                if (c == null) continue;
                var box = new VisualElement();
                box.style.marginBottom = 8;
                box.Add(ToggleRow(c.DisplayName + "  (" + c.preprocessor + ")",
                    () => idx < _c.controlNets.Count && _c.controlNets[idx].enabled,
                    v => { if (idx < _c.controlNets.Count) _c.controlNets[idx].enabled = v; }));
                box.Add(SliderRow("Strength", 0f, 2f,
                    () => idx < _c.controlNets.Count ? _c.controlNets[idx].scale : 0f,
                    v => { if (idx < _c.controlNets.Count) _c.controlNets[idx].scale = v; }, "0.00"));
                if (c.preprocessor == "canny")
                {
                    box.Add(SliderRow("Canny low threshold", 0, 255,
                        () => idx < _c.controlNets.Count ? _c.controlNets[idx].cannyLow : 0,
                        v => { if (idx < _c.controlNets.Count) _c.controlNets[idx].cannyLow = Mathf.RoundToInt(v); }, "0"));
                    box.Add(SliderRow("Canny high threshold", 0, 255,
                        () => idx < _c.controlNets.Count ? _c.controlNets[idx].cannyHigh : 0,
                        v => { if (idx < _c.controlNets.Count) _c.controlNets[idx].cannyHigh = Mathf.RoundToInt(v); }, "0"));
                }
                _cnLiveRows.Add(box);
            }
            DesignSystemBehaviourBase<UIDocument>.EnsureToggleKnobs(_cnLiveRows);
        }

        void PollControlNets()
        {
            if (_cnSetRows == null) return;
            string key = CnRowsKey();
            if (key != _cnSetBuilt) { _cnSetBuilt = key; RebuildCnSetRows(); }
            bool changed = _c.LoadedControlNetSetKey != null && _c.LoadedControlNetSetKey != _c.ControlNetSetKey();
            _cnSetNote.text = changed ? "ControlNet set changed: press 'Apply and reload model' below to use it." : "";
            _cnSetNote.style.color = changed ? (Color)new Color32(245, 180, 60, 255) : Muted;
        }

        // =============================================================== FX page
        VisualElement _fxPreRows, _fxPostRows;
        string _fxPreBuilt, _fxPostBuilt;
        DropdownField _fxPreDrop, _fxPostDrop;

        VisualElement BuildFxPage()
        {
            var page = Page();
            page.Add(FxSection("BEFORE THE MODEL (the input picture)",
                "Effects applied to the picture the model receives. Order matters, top to bottom. Live: change anything " +
                "while streaming, no reload.", true, out _fxPreRows, out _fxPreDrop));
            page.Add(FxSection("AFTER THE MODEL (the result)",
                "Effects applied to the generated picture, in order, before it is shown and sent out (NDI included).",
                false, out _fxPostRows, out _fxPostDrop));
            _binders.Add(() =>
            {
                string a = FxKey(_c.fxPre), b = FxKey(_c.fxPost);
                if (a != _fxPreBuilt) { _fxPreBuilt = a; RebuildFxRows(_fxPreRows, _c.fxPre, true); }
                if (b != _fxPostBuilt) { _fxPostBuilt = b; RebuildFxRows(_fxPostRows, _c.fxPost, false); }
            });
            return page;
        }

        VisualElement FxSection(string title, string caption, bool pre, out VisualElement rows, out DropdownField drop)
        {
            var sec = Section(title);
            sec.Add(Caption(caption));
            var labels = new List<string>();
            foreach (var d in FxCatalog.All) if (pre || !d.preOnly) labels.Add(d.label + (d.preOnly ? " (input only)" : ""));
            drop = new DropdownField(labels, 0);
            drop.AddToClassList("ds-dropdown");
            drop.style.flexGrow = 1; drop.style.marginRight = 6;
            var dd = drop;
            var row = Row(sec);
            row.Add(drop);
            row.Add(Btn("Add effect", "secondary", () =>
            {
                var avail = new List<FxCatalog.Def>();
                foreach (var d in FxCatalog.All) if (pre || !d.preOnly) avail.Add(d);
                int i = dd.index;
                if (i < 0 || i >= avail.Count) return;
                (pre ? _c.fxPre : _c.fxPost).Add(new FxEntry(avail[i].type));
            }));
            rows = new VisualElement();
            rows.style.marginTop = 8;
            sec.Add(rows);
            return sec;
        }

        static string FxKey(List<FxEntry> list)
        {
            var sb = new System.Text.StringBuilder();
            if (list != null) foreach (var e in list) sb.Append(e?.type).Append(';');
            return sb.ToString();
        }

        void RebuildFxRows(VisualElement host, List<FxEntry> list, bool pre)
        {
            host.Clear();
            if (list == null) return;
            for (int i = 0; i < list.Count; i++)
            {
                int idx = i;
                var e = list[i];
                if (e == null) continue;
                var def = FxCatalog.Find(e.type);
                e.EnsureDefaults();

                var card = new VisualElement();
                card.style.marginBottom = 10;
                card.style.paddingBottom = 6;
                card.style.borderBottomWidth = 1;
                card.style.borderBottomColor = (Color)new Color32(42, 42, 42, 255);

                var head = Row(card);
                var title = new Label((idx + 1) + ". " + (def != null ? def.label : e.type));
                title.AddToClassList("ds-body-2");
                title.style.flexGrow = 1;
                title.style.unityFontStyleAndWeight = FontStyle.Bold;
                head.Add(title);
                var up = Btn("Up", "ghost", () =>
                {
                    if (idx > 0 && idx < list.Count) { var t = list[idx]; list[idx] = list[idx - 1]; list[idx - 1] = t; }
                });
                up.style.marginRight = 4;
                var down = Btn("Down", "ghost", () =>
                {
                    if (idx >= 0 && idx < list.Count - 1) { var t = list[idx]; list[idx] = list[idx + 1]; list[idx + 1] = t; }
                });
                down.style.marginRight = 4;
                head.Add(up); head.Add(down);
                head.Add(Btn("Remove", "ghost", () => { if (idx < list.Count) list.RemoveAt(idx); }));

                if (def != null && !string.IsNullOrEmpty(def.description)) card.Add(Caption(def.description));
                card.Add(ToggleRow("Enabled", () => idx < list.Count && list[idx].enabled,
                    v => { if (idx < list.Count) list[idx].enabled = v; }));
                if (def != null)
                {
                    foreach (var p in def.parameters)
                    {
                        var pp = p;
                        string fmt = pp.integer ? "0" : "0.00";
                        card.Add(SliderRow(pp.label, pp.min, pp.max,
                            () => idx < list.Count ? list[idx].Get(pp.name, pp.defaultValue) : pp.defaultValue,
                            v =>
                            {
                                if (idx >= list.Count) return;
                                if (pp.integer) v = Mathf.Round(v);
                                if (pp.odd) { int iv = Mathf.RoundToInt(v); if (iv % 2 == 0) iv++; v = iv; }
                                list[idx].Set(pp.name, v);
                            }, fmt));
                    }
                }
                host.Add(card);
            }
            // toggles created after start-up need their knobs added too
            DesignSystemBehaviourBase<UIDocument>.EnsureToggleKnobs(host);
        }

        // ---- Windows GPU timeout ----
        Label _gpuTimeoutLabel, _gpuTimeoutNote;
        Button _gpuTimeoutBtn;
        bool _gpuTimeoutBusy;

        async void OnFixGpuTimeout()
        {
            if (_gpuTimeoutBusy) return;
            _gpuTimeoutBusy = true;
            try
            {
                string message = await _c.FixGpuTimeoutAsync();
                if (_gpuTimeoutNote != null) _gpuTimeoutNote.text = message;
            }
            finally { _gpuTimeoutBusy = false; }
        }

        // ---- model picker ----
        static readonly string[] KnownModels =
        {
            "stabilityai/sdxl-turbo", "stabilityai/sd-turbo", "Lykon/dreamshaper-8",
        };
        static readonly string[] KnownModelLabels =
        {
            "SDXL-Turbo (default, 1 step)", "SD-Turbo (SD 2.1, fast and light)", "DreamShaper 8 (SD 1.5, multi-step)",
        };
        DropdownField _modelDrop;
        Label _modelFolderLabel;
        readonly List<string> _modelIds = new List<string>();   // modelId behind each dropdown entry ("" = custom)
        bool _modelRefreshing;

        async void RefreshModelList()
        {
            if (_modelRefreshing) return;
            _modelRefreshing = true;
            try
            {
                var files = await _c.RefreshModelsAsync();
                var labels = new List<string>();
                _modelIds.Clear();
                labels.Add("(custom id below)"); _modelIds.Add("");
                for (int i = 0; i < KnownModels.Length; i++) { labels.Add(KnownModelLabels[i]); _modelIds.Add(KnownModels[i]); }
                foreach (var f in files)
                {
                    labels.Add($"Local: {f.name}  ({f.sizeMb:0} MB)");
                    _modelIds.Add(f.path);
                }
                _modelDropBuilding = true;
                _modelDrop.choices = labels;
                _modelDropBuilding = false;
                _modelFolderLabel.text = string.IsNullOrEmpty(_c.ModelFolder) ? "" : "Local models folder: " + _c.ModelFolder;
                SyncModelDropdown();
            }
            catch (Exception ex) { _modelFolderLabel.text = "Could not list models: " + ex.Message; }
            finally { _modelRefreshing = false; _modelDropBuilding = false; }
        }

        bool _modelDropBuilding;

        void SyncModelDropdown()
        {
            if (_modelDrop == null || _modelIds.Count == 0) return;
            string cur = (_c.modelId ?? "").Replace('\\', '/');
            int idx = 0;
            for (int i = 1; i < _modelIds.Count; i++)
                if (string.Equals(_modelIds[i], cur, StringComparison.OrdinalIgnoreCase)) { idx = i; break; }
            if (_modelDrop.index != idx && idx < _modelDrop.choices.Count()) _modelDrop.SetValueWithoutNotify(_modelDrop.choices.ElementAt(idx));
        }

        void OnModelPicked()
        {
            if (_modelDropBuilding) return;
            int i = _modelDrop.index;
            if (i > 0 && i < _modelIds.Count) _c.modelId = _modelIds[i];
        }

        DropdownField EnumDropdown(string[] labels, Func<int> get, Action<int> set)
        {
            var list = new List<string>(labels);
            var d = new DropdownField(list, Mathf.Clamp(get(), 0, list.Count - 1));
            d.AddToClassList("ds-dropdown");
            d.style.marginBottom = 8;
            d.RegisterValueChangedCallback(_ => { if (d.index >= 0) set(d.index); });
            _binders.Add(() =>
            {
                int i = Mathf.Clamp(get(), 0, list.Count - 1);
                if (d.index != i) d.SetValueWithoutNotify(list[i]);
            });
            return d;
        }

        VisualElement _loraWeightRows, _loraSetRows;
        DropdownField _loraDrop;
        Label _loraFolderLabel, _loraSetNote;
        string _loraWeightsBuilt, _loraSetBuilt;
        bool _loraListed, _loraListing, _loraListedConnected;

        string LoraRowsKey()
        {
            var sb = new System.Text.StringBuilder();
            if (_c.loras != null) foreach (var l in _c.loras) sb.Append(l?.path).Append(';');
            return sb.ToString();
        }

        void RebuildLoraWeightRows()
        {
            _loraWeightRows.Clear();
            if (_c.loras == null) return;
            for (int i = 0; i < _c.loras.Count; i++)
            {
                int idx = i;
                var l = _c.loras[i];
                if (l == null) continue;
                _loraWeightRows.Add(SliderRow(l.DisplayName, -1f, 2f,
                    () => idx < _c.loras.Count ? _c.loras[idx].weight : 0f,
                    v => { if (idx < _c.loras.Count) _c.loras[idx].weight = v; }, "0.00"));
            }
        }

        void RebuildLoraSetRows()
        {
            _loraSetRows.Clear();
            for (int i = 0; i < _c.loras.Count; i++)
            {
                int idx = i;
                var row = Row(_loraSetRows);
                row.style.marginBottom = 4;
                var name = new Label(_c.loras[i].DisplayName);
                name.AddToClassList("ds-body-2");
                name.style.flexGrow = 1;
                name.tooltip = _c.loras[i].path;
                row.Add(name);
                row.Add(Btn("Remove", "ghost", () => { if (idx < _c.loras.Count) _c.loras.RemoveAt(idx); }));
            }
        }

        async void RefreshLoraList()
        {
            if (_loraListing) return;
            _loraListing = true;
            _loraListed = true;      // one attempt per state change; the Refresh button forces another
            try
            {
                var files = await _c.RefreshLorasAsync();
                var names = new List<string>();
                foreach (var f in files) names.Add($"{f.name}  ({f.sizeMb:0} MB)");
                if (names.Count == 0) names.Add("(none found)");
                _loraDrop.choices = names;
                _loraDrop.index = 0;
                _loraFolderLabel.text = string.IsNullOrEmpty(_c.LoraFolder) ? "" : "Folder: " + _c.LoraFolder;
            }
            catch (Exception ex) { _loraFolderLabel.text = "Could not list LoRAs: " + ex.Message; }
            finally { _loraListing = false; }
        }

        void OnAddLora()
        {
            var files = _c.AvailableLoras;
            int i = _loraDrop.index;
            if (files == null || i < 0 || i >= files.Length) return;
            foreach (var l in _c.loras) if (l != null && l.path == files[i].path) return;   // already added
            _c.loras.Add(new LoraEntry(files[i].path, 1f));
        }

        void PollLoras()
        {
            if (_loraSetRows == null) return;
            bool connected = _c.ServerInfo != null && _c.Phase != DiffusionPhase.Disconnected && _c.Phase != DiffusionPhase.Error
                             && _c.Phase != DiffusionPhase.StartingServer;
            // list from the server when connected, straight from disk otherwise; again when that changes
            if (_loraListedConnected != connected) { _loraListed = false; _loraListedConnected = connected; }
            if (!_loraListed) RefreshLoraList();

            string key = LoraRowsKey();
            if (key != _loraSetBuilt) { _loraSetBuilt = key; RebuildLoraSetRows(); }

            bool loaded = _c.LoadedLoraSetKey != null;
            bool changed = loaded && _c.LoadedLoraSetKey != _c.LoraSetKey();
            _loraSetNote.text = changed ? "LoRA set changed: press 'Apply and reload model' below to use it." : "";
            _loraSetNote.style.color = changed ? (Color)new Color32(245, 180, 60, 255) : Muted;
        }

        // engine cache status on the Model page (re-checked, debounced, whenever the setup changes)
        Label _engineState;
        string _engineKey;
        float _engineQueryAt;
        bool _engineQuerying;

        void PollEngineStatus()
        {
            if (_engineState == null) return;
            bool connected = _c.ServerInfo != null &&
                (_c.Phase == DiffusionPhase.Connected || _c.Phase == DiffusionPhase.Loading || _c.Phase == DiffusionPhase.Ready ||
                 _c.Phase == DiffusionPhase.Running || _c.Phase == DiffusionPhase.AwaitingBuildConfirmation);
            if (!connected) { _engineState.text = "connect to check"; _engineKey = null; return; }

            string key = $"{_c.modelId}|{_c.width}x{_c.height}|{(_c.tIndexList != null ? _c.tIndexList.Length : 1)}|{_c.useTinyVae}|{_c.acceleration}|{_c.V2VKey()}|{_c.mode}|{_c.cfgType}";
            if (key != _engineKey)
            {
                _engineKey = key;
                _engineQueryAt = Time.unscaledTime + 0.5f;
                _engineState.text = "checking...";
            }
            if (_engineQueryAt > 0f && Time.unscaledTime >= _engineQueryAt && !_engineQuerying)
            {
                _engineQueryAt = 0f;
                QueryEngines(key);
            }
        }

        async void QueryEngines(string key)
        {
            _engineQuerying = true;
            try
            {
                var st = await _c.QueryEngineStatusAsync();
                if (key == _engineKey && st != null)
                    _engineState.text = st.cached
                        ? "cached (loads instantly)"
                        : $"will build: {st.MissingText} (about {st.estimateMinutes:0} min)";
            }
            catch { if (key == _engineKey) _engineState.text = "unknown"; }
            finally { _engineQuerying = false; }
        }

        async void OnReload()
        {
            if (_busy) return;
            _busy = true;
            try { await _c.ReloadModelAsync(); }
            catch (Exception ex) { _status.text = "Error: " + ex.Message; }
            finally { _busy = false; Refresh(); }
        }

        // =============================================================== NDI page
        Label _ndiState, _ndiConn, _ndiFrames, _ndiNote;

        VisualElement BuildNdiPage()
        {
            var page = Page();
            var s = Section("NDI OUTPUT");
            s.Add(Caption("Publish the generated stream as an NDI source so Resolume, OBS, vMix and other NDI software " +
                          "can pick it up, on this computer or over the network. It is sent from the server at the " +
                          "model's resolution."));
            s.Add(ToggleRow("Publish as NDI source", () => _c.ndiOutputEnabled, v => _c.ndiOutputEnabled = v));
            s.Add(Caption("Off by default. While off nothing is created or copied, so it costs no GPU, CPU or network."));

            var nameCap = Caption("Source name");
            nameCap.style.marginTop = 6;
            s.Add(nameCap);
            s.Add(Input(() => _c.ndiOutputName, v => _c.ndiOutputName = v));
            page.Add(s);

            var st = Section("STATUS");
            st.Add(InfoRow("State", out _ndiState));
            st.Add(InfoRow("Receivers connected", out _ndiConn));
            st.Add(InfoRow("Frames sent", out _ndiFrames));
            _ndiNote = Caption("");
            _ndiNote.style.marginTop = 6;
            st.Add(_ndiNote);
            page.Add(st);

            var help = Section("REQUIREMENTS");
            help.Add(Caption("Needs the free NDI Runtime (part of NDI Tools) installed on this computer. " +
                             "Other software should find the source named above in its NDI input list."));
            page.Add(help);

            _binders.Add(() =>
            {
                string state;
                if (!_c.NdiAvailable) state = "NDI runtime not available";
                else if (_c.NdiLive) state = "Live";
                else if (_c.ndiOutputEnabled) state = "Starting...";
                else state = "Off";
                _ndiState.text = state;
                _ndiConn.text = _c.NdiLive ? _c.NdiConnections.ToString() : "-";
                _ndiFrames.text = _c.NdiLive ? _c.NdiFrames.ToString() : "-";
                _ndiNote.text = !_c.NdiAvailable
                    ? "The server cannot create NDI senders: " + (_c.NdiUnavailableReason ?? "unknown reason")
                    : (!string.IsNullOrEmpty(_c.NdiError) ? "Last error: " + _c.NdiError : "");
            });
            return page;
        }

        // =============================================================== PRESETS page
        VisualElement BuildPresetsPage()
        {
            var page = Page();
            var s = Section("PRESETS");
            s.Add(Caption("A preset stores prompts, blends, generation, performance and model settings."));

            _presetDrop = new DropdownField(new List<string>(), 0);
            _presetDrop.AddToClassList("ds-dropdown");
            _presetDrop.style.marginBottom = 8;
            s.Add(_presetDrop);

            var row = Row(s);
            row.style.marginBottom = 8;
            var load = Btn("Load", "primary", OnPresetLoad); load.style.marginRight = 6;
            var del = Btn("Delete", "ghost", OnPresetDelete); del.style.marginRight = 6;
            var refresh = Btn("Refresh list", "ghost", RefreshPresetList);
            row.Add(load); row.Add(del); row.Add(refresh);

            s.Add(Caption("Save current settings as"));
            var saveRow = Row(s);
            _presetName = new TextField { value = "my preset" };
            _presetName.AddToClassList("ds-input");
            _presetName.style.flexGrow = 1; _presetName.style.marginRight = 6;
            saveRow.Add(_presetName);
            saveRow.Add(Btn("Save", "secondary", OnPresetSave));

            _presetStatus = Caption("");
            _presetStatus.style.marginTop = 8;
            s.Add(_presetStatus);
            s.Add(Caption("Folder: " + DiffusionPresetStore.Directory));
            page.Add(s);
            RefreshPresetList();
            return page;
        }

        void RefreshPresetList()
        {
            var names = new List<string>(DiffusionPresetStore.List());
            _presetDrop.choices = names;
            if (names.Count > 0) _presetDrop.SetValueWithoutNotify(names[0]);
            else _presetDrop.SetValueWithoutNotify("");
        }

        void OnPresetSave()
        {
            try
            {
                DiffusionPresetStore.Save(_presetName.value, _c.CapturePreset());
                _presetStatus.text = $"Saved '{_presetName.value}'.";
                RefreshPresetList();
                _presetDrop.SetValueWithoutNotify(_presetName.value);
            }
            catch (Exception ex) { _presetStatus.text = "Could not save: " + ex.Message; }
        }

        void OnPresetLoad()
        {
            var name = _presetDrop.value;
            if (string.IsNullOrEmpty(name)) { _presetStatus.text = "No preset selected."; return; }
            var p = DiffusionPresetStore.Load(name);
            if (p == null) { _presetStatus.text = $"'{name}' not found."; return; }
            _c.ApplyPreset(p);
            _presetStatus.text = $"Loaded '{name}'. Live settings applied; model/resolution need 'Apply and reload model'.";
        }

        void OnPresetDelete()
        {
            var name = _presetDrop.value;
            if (string.IsNullOrEmpty(name)) return;
            _presetStatus.text = DiffusionPresetStore.Delete(name) ? $"Deleted '{name}'." : $"'{name}' not found.";
            RefreshPresetList();
        }

        // =============================================================== ENVIRONMENT page
        VisualElement BuildEnvironmentPage()
        {
            var page = Page();

            var conn = Section("CONNECTION");
            var row = Row(conn);
            var host = Input(() => _c.host, v => _c.host = v);
            host.style.flexGrow = 1; host.style.marginRight = 6;
            var port = IntInput(() => _c.port, v => _c.port = v);
            port.style.width = 90;
            row.Add(host); row.Add(port);
            row.style.marginBottom = 8;
            conn.Add(ToggleRow("Start local server", () => _c.launchLocalServer, v => _c.launchLocalServer = v));
            page.Add(conn);

            var gt = Section("WINDOWS GPU TIMEOUT");
            _gpuTimeoutLabel = new Label("");
            _gpuTimeoutLabel.AddToClassList("ds-body-2");
            _gpuTimeoutLabel.style.whiteSpace = WhiteSpace.Normal;
            _gpuTimeoutLabel.style.marginBottom = 6;
            gt.Add(_gpuTimeoutLabel);
            _gpuTimeoutBtn = Btn("Fix GPU timeout (asks for administrator permission)", "secondary", OnFixGpuTimeout);
            _gpuTimeoutBtn.AddToClassList("ds-btn--block");
            gt.Add(_gpuTimeoutBtn);
            _gpuTimeoutNote = Caption("");
            _gpuTimeoutNote.style.marginTop = 6;
            gt.Add(_gpuTimeoutNote);
            gt.Add(Caption("Or run once in an administrator command prompt, then restart Windows:" + Environment.NewLine +
                           GpuTimeoutFix.ManualCommands));
            _binders.Add(() =>
            {
                var st = _c.GpuTimeout;
                gt.style.display = st == GpuTimeoutState.Unsupported ? DisplayStyle.None : DisplayStyle.Flex;
                GpuTimeoutFix.GetState(_c.GetInstaller().Paths.Base, out var secs);
                _gpuTimeoutLabel.text = GpuTimeoutFix.Describe(st, secs);
                _gpuTimeoutBtn.style.display = st == GpuTimeoutState.Short ? DisplayStyle.Flex : DisplayStyle.None;
            });
            page.Add(gt);

            var s = Section("PYTHON ENVIRONMENT");
            _envState = new Label("checking...");
            _envState.AddToClassList("ds-body-2");
            _envState.style.whiteSpace = WhiteSpace.Normal;
            _envState.style.marginBottom = 6;
            s.Add(_envState);
            s.Add(ToggleRow("Allow downloading Python / git if missing", () => _c.allowBootstrapDownloads,
                v => _c.allowBootstrapDownloads = v));

            _installBtn = Btn("Install environment", "secondary", OnInstall);
            _installBtn.AddToClassList("ds-btn--block");
            _installBtn.style.marginTop = 10; _installBtn.style.marginBottom = 8;
            s.Add(_installBtn);

            _installBar = new ProgressBar { lowValue = 0, highValue = 100 };
            _installBar.AddToClassList("ds-progress");
            _installBar.style.display = DisplayStyle.None;
            s.Add(_installBar);

            _logBox = new Label("");
            _logBox.AddToClassList("ds-caption");
            _logBox.style.whiteSpace = WhiteSpace.Normal;
            _logBox.style.color = Muted;
            _logBox.style.marginTop = 6;
            _logBox.style.display = DisplayStyle.None;
            s.Add(_logBox);
            page.Add(s);

            var srv = Section("SERVER LOG");
            _serverLog = new Label("(no output yet)");
            _serverLog.AddToClassList("ds-caption");
            _serverLog.style.whiteSpace = WhiteSpace.Normal;
            _serverLog.style.color = Muted;
            srv.Add(_serverLog);
            page.Add(srv);
            return page;
        }

        struct IpChoice { public string name, model, encoder; }

        static readonly IpChoice[] IpAdapterChoices =
        {
            new IpChoice { name = "SDXL - regular", model = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", encoder = "h94/IP-Adapter/sdxl_models/image_encoder" },
            new IpChoice { name = "SDXL - plus (closer to the reference)", model = "h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", encoder = "h94/IP-Adapter/models/image_encoder" },
            new IpChoice { name = "SDXL - FaceID (identity)", model = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin", encoder = "h94/IP-Adapter/models/image_encoder" },
            new IpChoice { name = "SD 1.5 - regular", model = "h94/IP-Adapter/models/ip-adapter_sd15.bin", encoder = "h94/IP-Adapter/models/image_encoder" },
            new IpChoice { name = "SD 1.5 - plus (closer to the reference)", model = "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors", encoder = "h94/IP-Adapter/models/image_encoder" },
            new IpChoice { name = "SD 1.5 - FaceID (identity)", model = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin", encoder = "h94/IP-Adapter/models/image_encoder" },
        };

        static string[] IpAdapterNames()
        {
            var n = new string[IpAdapterChoices.Length];
            for (int i = 0; i < n.Length; i++) n[i] = IpAdapterChoices[i].name;
            return n;
        }

        float _installStartedAt = -1f;

        static string FormatElapsed(float seconds)
        {
            int s = Mathf.Max(0, (int)seconds);
            return (s / 60) + ":" + (s % 60).ToString("00");
        }

        async void OnInstall()
        {
            if (_c.GetInstaller().IsRunning) { _c.CancelInstall(); return; }
            var inst = _c.GetInstaller();
            if (!ReferenceEquals(_boundInstaller, inst))   // the controller rebuilds it when the base folder changes
            {
                _boundInstaller = inst;
                inst.LogLine += l => _logQ.Enqueue(l);
                inst.Progress += (f, m) => { _installFraction = f; _installMessage = m; };
            }
            _installBar.style.display = DisplayStyle.Flex;
            _logBox.style.display = DisplayStyle.Flex;
            await _c.InstallEnvironmentAsync();
            Refresh();
        }

        // =============================================================== ABOUT page
        readonly Dictionary<string, Label> _about = new Dictionary<string, Label>();

        VisualElement BuildAboutPage()
        {
            var page = Page();
            var s = Section("ABOUT");
            void Info(string key, string label)
            {
                s.Add(InfoRow(label, out var v));
                v.style.whiteSpace = WhiteSpace.Normal;
                _about[key] = v;
            }
            Info("app", "Application");
            Info("server", "Server");
            Info("gpu", "GPU");
            Info("source", "Install source");
            Info("base", "Base folder");
            Info("logs", "Install logs");
            Info("presets", "Presets folder");
            page.Add(s);
            return page;
        }

        void RefreshAbout()
        {
            if (_about.Count == 0) return;
            var inst = _c.GetInstaller();
            var st = inst.Settings;
            _about["app"].text = $"Unity {Application.unityVersion}";
            string srv = "not connected";
            if (_c.ServerInfo != null)
                srv = $"v{(string)_c.ServerInfo["server_version"]}, protocol {(string)_c.ServerInfo["protocol"]}";
            _about["server"].text = srv;
            _about["gpu"].text = _c.ServerInfo != null
                ? $"{(string)_c.ServerInfo["gpu"]} ({(string)_c.ServerInfo["vram_total_gb"]} GB)" : "-";
            _about["source"].text = $"{st.repoUrl}\n{st.repoBranch} @ {st.repoCommit.Substring(0, 7)}";
            _about["base"].text = inst.Paths.Base;
            _about["logs"].text = System.IO.Path.Combine(inst.Paths.Base, "logs");
            _about["presets"].text = DiffusionPresetStore.Directory;
        }

        // =============================================================== OUTPUT card (always visible)
        VisualElement BuildOutput()
        {
            var s = Section("OUTPUT");
            s.style.width = 560;
            s.style.flexShrink = 0;
            s.style.marginLeft = 14;

            _startBtn = Btn("Connect & Start", "primary", OnStartStop);
            _startBtn.AddToClassList("ds-btn--block");
            _startBtn.style.marginBottom = 6;
            s.Add(_startBtn);
            _status = Caption("");
            s.Add(_status);

            var previews = new VisualElement();
            previews.style.flexGrow = 1;
            previews.style.flexShrink = 1;
            previews.style.minHeight = 180;
            previews.style.marginBottom = 10;

            if (showInputPreview)
            {
                var inCap = Caption("Input (sent to the model)");
                inCap.style.marginBottom = 2;
                previews.Add(inCap);
                _inputPreview = new Image { scaleMode = ScaleMode.ScaleToFit };
                _inputPreview.style.flexGrow = 1;
                _inputPreview.style.flexBasis = 0;
                _inputPreview.style.minHeight = 80;
                _inputPreview.style.marginBottom = 8;
                previews.Add(_inputPreview);
            }

            if (showInputPreview)
            {
                _controlPreviewCap = Caption("Control image (sent to the ControlNets)");
                _controlPreviewCap.style.marginBottom = 2;
                _controlPreviewCap.style.display = DisplayStyle.None;
                previews.Add(_controlPreviewCap);
                _controlPreview = new Image { scaleMode = ScaleMode.ScaleToFit };
                _controlPreview.style.flexGrow = 1;
                _controlPreview.style.flexBasis = 0;
                _controlPreview.style.minHeight = 80;
                _controlPreview.style.marginBottom = 8;
                _controlPreview.style.display = DisplayStyle.None;
                previews.Add(_controlPreview);
            }

            if (showInputPreview)
            {
                _referencePreviewCap = Caption("Reference image (IP-Adapter)");
                _referencePreviewCap.style.marginBottom = 2;
                _referencePreviewCap.style.display = DisplayStyle.None;
                previews.Add(_referencePreviewCap);
                _referencePreview = new Image { scaleMode = ScaleMode.ScaleToFit };
                _referencePreview.style.flexGrow = 1;
                _referencePreview.style.flexBasis = 0;
                _referencePreview.style.minHeight = 60;
                _referencePreview.style.marginBottom = 8;
                _referencePreview.style.display = DisplayStyle.None;
                previews.Add(_referencePreview);
            }

            if (showInputPreview)
            {
                _controlMapCap = Caption("Control map (what the ControlNet sees)");
                _controlMapCap.style.marginBottom = 2;
                _controlMapCap.style.display = DisplayStyle.None;
                previews.Add(_controlMapCap);
                _controlMapPreview = new Image { scaleMode = ScaleMode.ScaleToFit };
                _controlMapPreview.style.flexGrow = 1;
                _controlMapPreview.style.flexBasis = 0;
                _controlMapPreview.style.minHeight = 60;
                _controlMapPreview.style.marginBottom = 8;
                _controlMapPreview.style.display = DisplayStyle.None;
                previews.Add(_controlMapPreview);
            }

            var outCap = Caption("Result");
            outCap.style.marginBottom = 2;
            previews.Add(outCap);
            _preview = new Image { scaleMode = ScaleMode.ScaleToFit };
            _preview.style.flexGrow = 1;
            _preview.style.flexBasis = 0;
            _preview.style.minHeight = 80;
            previews.Add(_preview);
            s.Add(previews);

            s.Add(InfoRow("Frames per second", out var fps)); _stat["fps"] = fps;
            s.Add(InfoRow("Round-trip latency", out var lat)); _stat["lat"] = lat;
            s.Add(InfoRow("Inference", out var inf)); _stat["infer"] = inf;
            s.Add(InfoRow("Frames dropped (server)", out var drop)); _stat["drops"] = drop;

            s.Add(Caption("GPU memory"));
            var meter = new VisualElement();
            meter.AddToClassList("ds-meter"); meter.AddToClassList("ds-meter--secondary");
            _vramFill = new VisualElement();
            _vramFill.AddToClassList("ds-meter__fill");
            _vramFill.style.width = Length.Percent(0);
            _vramLabel = new Label("-");
            _vramLabel.AddToClassList("ds-meter__label");
            meter.Add(_vramFill); meter.Add(_vramLabel);
            meter.style.flexGrow = 1;
            meter.style.flexShrink = 1;
            // same corner radius as the status badge (6 px) instead of the design system's pill shape
            void Radius(VisualElement e, float r)
            {
                e.style.borderTopLeftRadius = r; e.style.borderTopRightRadius = r;
                e.style.borderBottomLeftRadius = r; e.style.borderBottomRightRadius = r;
            }
            Radius(meter, 6f);
            Radius(_vramFill, 5f);
            meter.style.height = 22;   // same height as the badge so they sit flush
            meter.style.minHeight = 22;

            // phase badge (Loading / Running / ...) sits beside the GPU memory bar
            _statusBadge = new Label("Disconnected");
            _statusBadge.AddToClassList("ds-badge");
            _statusBadge.AddToClassList("ds-badge--common");
            _statusBadge.style.marginLeft = 10;
            _statusBadge.style.flexShrink = 0;
            _statusBadge.style.marginTop = 0;      // the design system's badge margins would offset it by 1 px
            _statusBadge.style.marginBottom = 0;
            _statusBadge.style.height = 22;
            _statusBadge.style.minHeight = 22;
            var memRow = Row(s);
            memRow.Add(meter);
            memRow.Add(_statusBadge);
            return s;
        }

        async void OnStartStop()
        {
            if (_busy) return;
            _busy = true;
            try
            {
                if (_c.Phase == DiffusionPhase.Running) await _c.StopAsync();
                else if (_c.Phase == DiffusionPhase.Ready) await _c.StartAsync();
                else await _c.ConnectAndStartAsync();
            }
            catch (Exception ex) { if (_status != null) _status.text = "Error: " + ex.Message; }
            finally { _busy = false; Refresh(); }
        }

        // =============================================================== refresh / update
        void Refresh()
        {
            if (_c == null) return;
            if (_status != null) _status.text = _c.StatusMessage;
            if (_startBtn != null)
                _startBtn.text = _c.Phase == DiffusionPhase.Running ? "Stop"
                    : _c.Phase == DiffusionPhase.Ready ? "Start"
                    : _c.Phase == DiffusionPhase.AwaitingBuildConfirmation ? "Waiting for your answer above..."
                    : "Connect & Start";

            if (_statusBadge != null)
            {
                _statusBadge.text = _c.Phase.ToString();
                ApplyBadgeColors(_c.Phase, 0f);
            }
            RefreshAbout();
        }

        // ---- status badge colours --------------------------------------------------------------------
        // idle states are greys (Disconnected dark, Connected brighter, Connecting pulses between them),
        // busy = amber, Running = green, Error = red.
        static readonly Color BadgeDisconnected = new Color32(110, 110, 110, 255);
        static readonly Color BadgeConnected = new Color32(200, 200, 200, 255);
        static readonly Color BadgeReady = new Color32(236, 236, 236, 255);
        static readonly Color BadgeBusy = new Color32(245, 158, 11, 255);
        static readonly Color BadgeRunning = new Color32(82, 240, 86, 255);
        static readonly Color BadgeError = new Color32(239, 68, 68, 255);

        void ApplyBadgeColors(DiffusionPhase phase, float pulse)
        {
            Color c;
            float bg = 0.16f;
            switch (phase)
            {
                case DiffusionPhase.Disconnected: c = BadgeDisconnected; bg = 0.10f; break;
                case DiffusionPhase.Connecting: c = Color.Lerp(BadgeDisconnected, BadgeConnected, pulse); bg = Mathf.Lerp(0.10f, 0.16f, pulse); break;
                case DiffusionPhase.Connected: c = BadgeConnected; break;
                case DiffusionPhase.Ready: c = BadgeReady; bg = 0.20f; break;
                case DiffusionPhase.Running: c = BadgeRunning; break;
                case DiffusionPhase.Error: c = BadgeError; break;
                default: c = BadgeBusy; break;   // Installing, StartingServer, Loading, AwaitingBuildConfirmation
            }
            _statusBadge.style.color = c;
            _statusBadge.style.backgroundColor = new Color(c.r, c.g, c.b, bg);
        }

        void Update()
        {
            if (_c == null) return;
            if (_statusBadge != null && _c.Phase == DiffusionPhase.Connecting)
                ApplyBadgeColors(DiffusionPhase.Connecting, 0.5f - 0.5f * Mathf.Cos(Time.unscaledTime * 2f * Mathf.PI * 1.2f));
            for (int i = 0; i < _binders.Count; i++) _binders[i]();
            PumpLogs();
            if (_console != null)
            {
                _console.SetInstallProgress(_installFraction, _installMessage);
                _console.Update();
            }

            if (_preview != null && _c.ResultTexture != null && _preview.image != _c.ResultTexture)
                _preview.image = _c.ResultTexture;
            if (_inputPreview != null && _c.SendTexture != null && _inputPreview.image != _c.SendTexture)
                _inputPreview.image = _c.SendTexture;
            if (_controlMapPreview != null)
            {
                bool showMap = _c.ControlMapActive && _c.ControlMapTexture != null;
                var d3 = showMap ? DisplayStyle.Flex : DisplayStyle.None;
                if (_controlMapPreview.style.display != d3) { _controlMapPreview.style.display = d3; _controlMapCap.style.display = d3; }
                if (showMap && _controlMapPreview.image != _c.ControlMapTexture) _controlMapPreview.image = _c.ControlMapTexture;
            }
            if (_referencePreview != null)
            {
                bool showRef = _c.useIpAdapter && _c.referenceTexture != null && _c.ReferenceSendTexture != null;
                var d2 = showRef ? DisplayStyle.Flex : DisplayStyle.None;
                if (_referencePreview.style.display != d2) { _referencePreview.style.display = d2; _referencePreviewCap.style.display = d2; }
                if (showRef && _referencePreview.image != _c.ReferenceSendTexture) _referencePreview.image = _c.ReferenceSendTexture;
            }
            if (_controlPreview != null)
            {
                bool show = _c.ControlSeparateActive && _c.ControlSendTexture != null;
                var disp = show ? DisplayStyle.Flex : DisplayStyle.None;
                if (_controlPreview.style.display != disp) { _controlPreview.style.display = disp; _controlPreviewCap.style.display = disp; }
                if (show && _controlPreview.image != _c.ControlSendTexture) _controlPreview.image = _c.ControlSendTexture;
            }

            if (_c.Phase == DiffusionPhase.Running && _stat.Count > 0)
            {
                _stat["fps"].text = _c.ReceiveFps.ToString("0.0");
                _stat["lat"].text = _c.LastLatencyMs.ToString("0") + " ms";
                if (_c.Stats != null)
                {
                    _stat["infer"].text = ((float)_c.Stats["infer_ms"]).ToString("0.0") + " ms";
                    _stat["drops"].text = ((int)_c.Stats["queue_drops"]).ToString();
                    float used = (float)_c.Stats["vram_used_gb"], total = Mathf.Max(0.01f, (float)_c.Stats["vram_total_gb"]);
                    _vramFill.style.width = Length.Percent(Mathf.Clamp01(used / total) * 100f);
                    _vramLabel.text = $"{used:0.0} / {total:0.0} GB";
                }
            }
        }

        void PumpLogs()
        {
            // follow the installer however the install was started (button, command line, auto-install)
            {
                var inst0 = _c.GetInstaller();
                if (!ReferenceEquals(_boundInstaller, inst0))
                {
                    _boundInstaller = inst0;
                    inst0.LogLine += l => _logQ.Enqueue(l);
                    inst0.Progress += (f, m) => { _installFraction = f; _installMessage = m; };
                }
                if (inst0.IsRunning)
                {
                    if (_installStartedAt < 0f) _installStartedAt = Time.unscaledTime;
                    float live = inst0.LiveProgress(out string liveMsg);
                    _installFraction = live;
                    _installMessage = liveMsg + "  -  " + FormatElapsed(Time.unscaledTime - _installStartedAt);
                }
                else _installStartedAt = -1f;
            }

            if (_envState != null)
            {
                var inst = _c.GetInstaller();
                if (inst.IsRunning)
                {
                    _installBtn.text = "Cancel install";
                    _installBar.style.display = DisplayStyle.Flex;
                    _logBox.style.display = DisplayStyle.Flex;
                    if (_installFraction >= 0)
                    {
                        _installBar.value = _installFraction * 100f;
                        _installBar.title = _installMessage;
                    }
                }
                else
                {
                    _installBtn.text = _c.EnvironmentState == InstallState.Installed ? "Re-verify / repair" : "Install environment";
                }
                _envState.text = $"{_c.EnvironmentState}  -  {inst.Paths.Base}";
            }

            bool changed = false;
            while (_logQ.TryDequeue(out var line))
            {
                _logLines.Add(line.Length > 150 ? line.Substring(0, 150) : line);
                if (_logLines.Count > 10) _logLines.RemoveAt(0);
                changed = true;
            }
            if (changed && _logBox != null) _logBox.text = string.Join(Environment.NewLine, _logLines);

            changed = false;
            while (_serverQ.TryDequeue(out var line))
            {
                string trimmed = line.StartsWith("[server] ") ? line.Substring(9) : line;
                _serverLines.Add(trimmed.Length > 150 ? trimmed.Substring(0, 150) : trimmed);
                if (_serverLines.Count > 12) _serverLines.RemoveAt(0);
                changed = true;
            }
            if (changed && _serverLog != null) _serverLog.text = string.Join(Environment.NewLine, _serverLines);
        }
    }
}
