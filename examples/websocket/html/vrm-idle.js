/**
 * VRMIdle – Procedural idle animation, body motion, expression & blink for VRM models.
 *
 * Core animation has no DOM dependencies.
 * Call createInspector() to optionally generate a settings UI overlay.
 *
 * Usage:
 *   const idle = new VRMIdle({ isAudioPlaying: () => someFlag });
 *   idle.vrm = loadedVrmInstance;
 *   const inspector = idle.createInspector();           // optional
 *   inspector.addTab('Light', buildFn, { position: 0 }); // add custom tabs
 *   // Each frame:
 *   idle.update(delta);
 */
class VRMIdle {

    static DEG = Math.PI / 180;

    // --- Pose parameter schema ---
    static POSE_PARAMS = [
        { key: 'armSpread', label: 'Arm spread',  min: -60, max: 120, def: 60,  fmt: v => v + '\u00b0' },
        { key: 'armFwd',    label: 'Arm forward', min: -30, max: 60,  def: 0,   fmt: v => v + '\u00b0' },
        { key: 'elbow',     label: 'Elbow bend',  min: 0,   max: 60,  def: 7,   fmt: v => v + '\u00b0' },
        { key: 'breath',    label: 'Breath',      min: 0,   max: 500, def: 200, fmt: v => v + '%' },
    ];

    // --- Bones available for body motion ---
    static SWAY_BONES = [
        'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
        'leftShoulder', 'rightShoulder',
    ];

    // --- Per-bone parameter schema (slider integer → actual = value / div) ---
    static SWAY_PARAMS = [
        { key: 'globalAmp',   label: 'Amplitude',   min: 0,   max: 300, def: 100, div: 100, fmt: v => (v / 100).toFixed(2) },
        { key: 'yawAmp',      label: 'Yaw Amp',     min: 0,   max: 600, def: 125, div: 10,  fmt: v => (v / 10).toFixed(1) + '\u00b0' },
        { key: 'yawSpeed',    label: 'Yaw Speed',   min: 0,   max: 200, def: 30,  div: 100, fmt: v => (v / 100).toFixed(2) },
        { key: 'yawSmooth',   label: 'Yaw Smooth',  min: 5,   max: 200, def: 40,  div: 100, fmt: v => (v / 100).toFixed(2) + 's' },
        { key: 'rollAmp',     label: 'Roll Amp',     min: 0,   max: 300, def: 15,  div: 10,  fmt: v => (v / 10).toFixed(1) + '\u00b0' },
        { key: 'rollSpeed',   label: 'Roll Speed',   min: 0,   max: 200, def: 30,  div: 100, fmt: v => (v / 100).toFixed(2) },
        { key: 'rollSmooth',  label: 'Roll Smooth',  min: 5,   max: 200, def: 40,  div: 100, fmt: v => (v / 100).toFixed(2) + 's' },
        { key: 'octaves',     label: 'Octaves',      min: 1,   max: 6,   def: 2,   div: 1,   fmt: v => String(v) },
        { key: 'lacunarity',  label: 'Lacunarity',   min: 10,  max: 40,  def: 20,  div: 10,  fmt: v => (v / 10).toFixed(1) },
        { key: 'persistence', label: 'Persistence',  min: 10,  max: 100, def: 50,  div: 100, fmt: v => (v / 100).toFixed(2) },
        { key: 'jerkProb',    label: 'Jerk Prob',    min: 0,   max: 100, def: 0,   div: 1000, fmt: v => (v / 1000).toFixed(3) },
        { key: 'jerkCooldown', label: 'Jerk CD',     min: 5,   max: 50,  def: 15,  div: 10,  fmt: v => (v / 10).toFixed(1) + 's' },
        { key: 'jerkMultMin', label: 'Jerk Min',     min: 10,  max: 30,  def: 15,  div: 10,  fmt: v => (v / 10).toFixed(1) + 'x' },
        { key: 'jerkMultMax', label: 'Jerk Max',     min: 15,  max: 50,  def: 25,  div: 10,  fmt: v => (v / 10).toFixed(1) + 'x' },
        { key: 'convAmp',     label: 'Conv Amp',     min: 0,   max: 100, def: 20,  div: 100, fmt: v => (v / 100).toFixed(2) },
        { key: 'fadeTime',    label: 'Fade Time',    min: 10,  max: 200, def: 50,  div: 100, fmt: v => (v / 100).toFixed(2) + 's' },
    ];

    // --- VRM 0.x → 1.0 expression name mapping ---
    static EXPRESSION_MAP = {
        joy:        'happy',
        fun:        'relaxed',
        sorrow:     'sad',
        angry:      'angry',
        surprise:   'surprised',
        a:          'aa',
        i:          'ih',
        u:          'ou',
        e:          'ee',
        o:          'oh',
        blink_l:    'blinkLeft',
        blink_r:    'blinkRight',
        lookup:     'lookUp',
        lookdown:   'lookDown',
        lookleft:   'lookLeft',
        lookright:  'lookRight',
    };

    // --- Viseme names ---
    static VISEMES = ['aa', 'ih', 'ou', 'ee', 'oh'];

    static MOUTH_TO_VISEME = {
        closed: null,
        half: { aa: 0.4 },
        open: { aa: 1.0 },
        u: { ou: 0.8 },
        e: { ee: 0.7 },
    };

    // Viseme smoothing speed (higher = faster response)
    static VISEME_LERP_SPEED = 25;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * @param {Object} [options]
     * @param {Function} [options.isAudioPlaying] - Returns true when audio is playing (for conversation weight)
     */
    constructor({ isAudioPlaying } = {}) {
        this._vrm = null;
        this.isAudioPlaying = isAudioPlaying || (() => false);

        // Current face name (used by blink to decide whether to blink)
        this.currentFaceName = 'neutral';

        // Idle pose parameters (caller sets these, e.g. from sliders)
        this.pose = {
            armSpread: 60,
            armFwd: 0,
            elbow: 7,
            breath: 200,
        };

        // Body motion targets (managed via swayAdd/swayRemove/swayLoadData)
        this.swayTargets = [];
        this.swayEnabled = true;
        this._swayFaceForward = false;
        this.swayPauseWhen = null;    // callback → true で自動停止 (null = 機能OFF)
        this.swayResumeDelay = 20;    // 沈黙後の再開までの秒数
        this._swayAutoDisabled = false;
        this._swayIdleTimer = 0;

        // Viseme smoothing state
        this._visemeTarget = {};   // target values per viseme name
        this._visemeCurrent = {};  // current (smoothed) values per viseme name
        for (const v of VRMIdle.VISEMES) {
            this._visemeTarget[v] = 0;
            this._visemeCurrent[v] = 0;
        }

        // Expression auto-revert state
        this._exprTimeout = null;
        this._exprUpdateId = 0;

        // Blink state
        this._blinkPhase = 'idle';
        this._blinkValue = 0;
        this._scheduleNextBlink();

        // Elapsed time (managed internally)
        this._elapsedTime = 0;

        // Animation registry – caller populates, e.g.:
        //   idle.animations.wave = { label: 'Wave', fn: (dur) => { ... }, defaultDuration: 2 };
        this.animations = {};
    }

    get vrm() { return this._vrm; }
    set vrm(v) { this._vrm = v; }

    // ================================================================
    // Main update (call once per frame with raw delta from clock)
    // ================================================================
    update(delta) {
        delta = Math.min(delta, 0.1);
        this._elapsedTime += delta;
        if (!this._vrm) return;
        this._updateSwayAutoPause(delta);
        this._applyBasePose();
        this._updateBreathing(this._elapsedTime);
        this._updateSway(delta, this._elapsedTime);
        this._updateVisemes(delta);
        this._updateBlink(delta);
        this._vrm.update(delta);
    }

    // ================================================================
    // Expression Control
    // ================================================================
    _setExpr(name, value) {
        this._vrm?.expressionManager?.setValue(name, value);
    }

    clearVisemes() {
        for (const v of VRMIdle.VISEMES) {
            this._visemeTarget[v] = 0;
        }
    }

    /**
     * @param {string} faceName - Expression name (e.g. 'happy', 'neutral')
     * @param {number} [duration] - Seconds before auto-reverting to neutral. 0 or omitted = no revert.
     */
    applyExpression(faceName, duration) {
        faceName = (faceName || 'neutral').toLowerCase();
        faceName = VRMIdle.EXPRESSION_MAP[faceName] || faceName;
        if (this.currentFaceName && this.currentFaceName !== 'neutral') {
            this._setExpr(this.currentFaceName, 0);
        }
        this.currentFaceName = faceName;
        if (faceName !== 'neutral') {
            this._setExpr(faceName, 1.0);
        }

        if (this._exprTimeout) clearTimeout(this._exprTimeout);
        if (duration > 0) {
            const updateId = ++this._exprUpdateId;
            this._exprTimeout = setTimeout(() => {
                if (this._exprUpdateId === updateId) {
                    this.applyExpression('neutral');
                }
            }, duration * 1000);
        }
    }

    applyViseme(mouthShape) {
        // Set all targets to 0, then apply the mapping
        for (const v of VRMIdle.VISEMES) {
            this._visemeTarget[v] = 0;
        }
        const mapping = VRMIdle.MOUTH_TO_VISEME[mouthShape];
        if (mapping) {
            for (const [viseme, value] of Object.entries(mapping)) {
                this._visemeTarget[viseme] = value;
            }
        }
    }

    // ================================================================
    // Private: Viseme Smoothing
    // ================================================================
    _updateVisemes(delta) {
        const speed = VRMIdle.VISEME_LERP_SPEED;
        const t = Math.min(1, speed * delta);  // lerp factor clamped to [0,1]
        for (const v of VRMIdle.VISEMES) {
            const cur = this._visemeCurrent[v];
            const tgt = this._visemeTarget[v];
            if (cur !== tgt) {
                const next = cur + (tgt - cur) * t;
                // Snap to target when close enough to avoid endless tiny updates
                this._visemeCurrent[v] = Math.abs(next - tgt) < 0.005 ? tgt : next;
            }
            this._setExpr(v, this._visemeCurrent[v]);
        }
    }

    // ================================================================
    // Sway Management
    // ================================================================
    swayDefaultParams() {
        const p = {};
        for (const d of VRMIdle.SWAY_PARAMS) p[d.key] = d.def;
        return p;
    }

    _swayCreateState() {
        return {
            yaw: 0, roll: 0,
            yawVel: { v: 0 }, rollVel: { v: 0 },
            weight: 1, weightVel: { v: 0 },
            lastJerkTime: -10,
            seedYaw: Math.random() * 1000,
            seedRoll: Math.random() * 1000,
        };
    }

    _swayVal(target, key) {
        const pd = VRMIdle.SWAY_PARAMS.find(p => p.key === key);
        return target.params[key] / (pd ? pd.div : 1);
    }

    /**
     * Enable or disable all sway.
     * @param {boolean} enabled
     * @param {boolean} [faceForward=false] - When disabling, smoothly return to forward-facing pose.
     */
    setSwayEnabled(enabled, faceForward = false) {
        this.swayEnabled = enabled;
        this._swayFaceForward = !enabled && faceForward;
    }

    /** Add a bone target. Returns the created target object. */
    swayAdd(bone) {
        const target = { bone, params: this.swayDefaultParams(), state: this._swayCreateState() };
        this.swayTargets.push(target);
        return target;
    }

    /** Remove a bone target by index. */
    swayRemove(index) {
        this.swayTargets.splice(index, 1);
    }

    /** Reset to default (single upperChest target). */
    swayReset() {
        this.swayTargets.length = 0;
        this.swayAdd('spine');
    }

    /**
     * Load body motion data from a serialized array (e.g. from localStorage).
     * @param {Array|null} data - Array of { bone, params } objects, or null/empty for default.
     */
    swayLoadData(data) {
        this.swayTargets.length = 0;
        if (Array.isArray(data) && data.length > 0) {
            for (const s of data) {
                this.swayTargets.push({
                    bone: s.bone,
                    params: { ...this.swayDefaultParams(), ...s.params },
                    state: this._swayCreateState(),
                });
            }
        } else {
            this.swayAdd('spine');
        }
    }

    /**
     * Return serializable array for persistence (caller handles localStorage).
     * @returns {Array} Array of { bone, params } objects.
     */
    swaySaveData() {
        return this.swayTargets.map(t => ({ bone: t.bone, params: { ...t.params } }));
    }

    // ================================================================
    // Animation Dispatcher
    // ================================================================
    playAnimation(name, duration) {
        name = (name || '').toLowerCase();
        const anim = this.animations[name];
        if (anim) anim.fn(duration || anim.defaultDuration);
    }

    // ================================================================
    // Private: Base Pose (arms)
    // ================================================================
    _applyBasePose() {
        const vrm = this._vrm;
        const DEG = VRMIdle.DEG;

        const armZ = this.pose.armSpread * DEG;
        const armX = this.pose.armFwd * DEG;
        const elbowZ = this.pose.elbow * DEG;

        const lua = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
        const rua = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
        const lla = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
        const rla = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');

        if (lua) lua.rotation.set(armX, 0, armZ);
        if (rua) rua.rotation.set(armX, 0, -armZ);
        if (lla) lla.rotation.set(0, 0, elbowZ);
        if (rla) rla.rotation.set(0, 0, -elbowZ);
    }

    // ================================================================
    // Private: Breathing Motion
    // ================================================================
    _updateBreathing(t) {
        const vrm = this._vrm;
        const breathScale = this.pose.breath / 100;
        const breath = Math.sin(t * 1.5) * breathScale;

        // Shoulders (rise with inhale)
        const lsh = vrm.humanoid.getNormalizedBoneNode('leftShoulder');
        const rsh = vrm.humanoid.getNormalizedBoneNode('rightShoulder');
        if (lsh) lsh.rotation.set(0, 0, breath * 0.008);
        if (rsh) rsh.rotation.set(0, 0, -breath * 0.008);

        // Spine / chest (set all 3 axes so sway can safely overwrite Y/Z)
        const spine = vrm.humanoid.getNormalizedBoneNode('spine');
        if (spine) spine.rotation.set(breath * 0.015, 0, 0);
        const chest = vrm.humanoid.getNormalizedBoneNode('upperChest')
                   || vrm.humanoid.getNormalizedBoneNode('chest');
        if (chest) chest.rotation.set(breath * 0.01, 0, 0);
    }

    // ================================================================
    // Private: Noise & Smooth Utilities
    // ================================================================
    static _nhash(n) {
        const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
        return s - Math.floor(s);
    }

    static _noise1d(x) {
        const i = Math.floor(x);
        const f = x - i;
        const u = f * f * (3 - 2 * f);
        return VRMIdle._nhash(i) * (1 - u) + VRMIdle._nhash(i + 1) * u;
    }

    static _fbm(t, octaves, lacunarity, persistence) {
        let value = 0, amp = 1, freq = 1, maxAmp = 0;
        for (let i = 0; i < octaves; i++) {
            value += amp * (VRMIdle._noise1d(t * freq) * 2 - 1);
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }

    static _smoothDamp(current, target, vel, smoothTime, dt) {
        smoothTime = Math.max(0.0001, smoothTime);
        const w = 2 / smoothTime;
        const x = w * dt;
        const e = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
        const d = current - target;
        const tmp = (vel.v + w * d) * dt;
        vel.v = (vel.v - w * tmp) * e;
        let r = target + (d + tmp) * e;
        if ((target > current) === (r > target)) { r = target; vel.v = 0; }
        return r;
    }

    // ================================================================
    // Private: Auto-pause sway during conversation
    // ================================================================
    _updateSwayAutoPause(delta) {
        if (!this.swayPauseWhen) return;
        if (this.swayPauseWhen()) {
            this._swayIdleTimer = 0;
            if (this.swayEnabled) {
                this._swayAutoDisabled = true;
                this.setSwayEnabled(false, true);
            }
        } else if (this._swayAutoDisabled) {
            this._swayIdleTimer += delta;
            if (this._swayIdleTimer >= this.swayResumeDelay) {
                this._swayAutoDisabled = false;
                this.setSwayEnabled(true);
            }
        }
    }

    // ================================================================
    // Private: Sway (fBm + SmoothDamp + Jerk, multi-bone)
    // ================================================================
    _updateSway(delta, t) {
        if (delta <= 0) return;
        if (!this.swayEnabled && !this._swayFaceForward) return;
        const vrm = this._vrm;
        const DEG = VRMIdle.DEG;
        let allSettled = true;

        for (const target of this.swayTargets) {
            const node = vrm.humanoid.getNormalizedBoneNode(target.bone);
            if (!node) continue;
            const bi = target.state;
            const gAmp = this._swayVal(target, 'globalAmp');

            if (!this.swayEnabled) {
                // Smooth return to center
                bi.yaw = VRMIdle._smoothDamp(bi.yaw, 0, bi.yawVel, this._swayVal(target, 'yawSmooth'), delta);
                bi.roll = VRMIdle._smoothDamp(bi.roll, 0, bi.rollVel, this._swayVal(target, 'rollSmooth'), delta);
                const w = gAmp * bi.weight;
                node.rotation.y = bi.yaw * DEG * w;
                node.rotation.z = bi.roll * DEG * w;
                if (Math.abs(bi.yaw) > 0.01 || Math.abs(bi.roll) > 0.01) allSettled = false;
                continue;
            }

            if (gAmp < 0.001) continue;

            // Conversation weight
            const convAmp = this._swayVal(target, 'convAmp');
            const fadeTime = this._swayVal(target, 'fadeTime');
            const targetW = this.isAudioPlaying() ? convAmp : 1;
            bi.weight = VRMIdle._smoothDamp(bi.weight, targetW, bi.weightVel, fadeTime, delta);

            // fBm noise targets
            const oct = this._swayVal(target, 'octaves');
            const lac = this._swayVal(target, 'lacunarity');
            const pers = this._swayVal(target, 'persistence');
            let tgtYaw = VRMIdle._fbm(t * this._swayVal(target, 'yawSpeed') + bi.seedYaw, oct, lac, pers) * this._swayVal(target, 'yawAmp');
            let tgtRoll = VRMIdle._fbm(t * this._swayVal(target, 'rollSpeed') + bi.seedRoll, oct, lac, pers) * this._swayVal(target, 'rollAmp');

            // Jerk
            const jerkProb = this._swayVal(target, 'jerkProb');
            const jerkCD = this._swayVal(target, 'jerkCooldown');
            if (jerkProb > 0 && t - bi.lastJerkTime > jerkCD && Math.random() < jerkProb) {
                const jMin = this._swayVal(target, 'jerkMultMin');
                const jMax = this._swayVal(target, 'jerkMultMax');
                const jm = jMin + Math.random() * (jMax - jMin);
                tgtYaw *= -jm;
                tgtRoll *= -jm;
                bi.lastJerkTime = t;
            }

            // SmoothDamp
            bi.yaw = VRMIdle._smoothDamp(bi.yaw, tgtYaw, bi.yawVel, this._swayVal(target, 'yawSmooth'), delta);
            bi.roll = VRMIdle._smoothDamp(bi.roll, tgtRoll, bi.rollVel, this._swayVal(target, 'rollSmooth'), delta);

            // Apply rotation (set, not accumulate)
            const w = gAmp * bi.weight;
            node.rotation.y = bi.yaw * DEG * w;
            node.rotation.z = bi.roll * DEG * w;
        }

        if (!this.swayEnabled && allSettled) {
            this._swayFaceForward = false;
        }
    }

    // ================================================================
    // Private: Smooth Blink
    // ================================================================
    _scheduleNextBlink() {
        const interval = 3000 + Math.random() * 3000;
        setTimeout(() => {
            if (!this._vrm || (this.currentFaceName && this.currentFaceName !== 'neutral')) {
                this._scheduleNextBlink();
                return;
            }
            this._blinkPhase = 'closing';
        }, interval);
    }

    _updateBlink(delta) {
        if (this._blinkPhase === 'closing') {
            this._blinkValue = Math.min(1, this._blinkValue + 18 * delta);
            if (this._blinkValue >= 1) this._blinkPhase = 'opening';
        } else if (this._blinkPhase === 'opening') {
            this._blinkValue = Math.max(0, this._blinkValue - 12 * delta);
            if (this._blinkValue <= 0) {
                this._blinkValue = 0;
                this._blinkPhase = 'idle';
                this._scheduleNextBlink();
            }
        }
        this._setExpr('blink', this._blinkValue);
    }

    // ================================================================
    // Inspector UI (optional – call createInspector to enable)
    // ================================================================

    static _inspectorCSSInjected = false;

    static _injectInspectorCSS() {
        if (VRMIdle._inspectorCSSInjected) return;
        VRMIdle._inspectorCSSInjected = true;
        const style = document.createElement('style');
        style.textContent = `
.vrmi-toggle{position:fixed;top:10px;right:10px;z-index:1001;width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,0.65);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgba(0,0,0,0.08);box-shadow:0 2px 8px rgba(0,0,0,0.06);cursor:pointer;font-size:18px;display:flex;align-items:center;justify-content:center;padding:0;min-height:auto;margin:0;transition:background 0.2s}
.vrmi-toggle:hover{background:rgba(255,255,255,0.85)}
.vrmi-panel{position:fixed;top:0;right:0;width:280px;height:100vh;background:rgba(245,245,247,0.72);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);box-shadow:-1px 0 24px rgba(0,0,0,0.08);z-index:1000;display:flex;flex-direction:column;transform:translateX(100%);transition:transform 0.3s cubic-bezier(0.4,0,0.2,1)}
.vrmi-panel.open{transform:translateX(0)}
.vrmi-header{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid rgba(0,0,0,0.06);font-weight:600;color:#1d1d1f;font-size:14px;flex-shrink:0;letter-spacing:-0.01em}
.vrmi-close{all:unset;cursor:pointer;font-size:20px;color:#86868b;padding:0 4px;line-height:1;transition:color 0.2s}
.vrmi-close:hover{color:#1d1d1f}
.vrmi-tabs{display:flex;border-bottom:1px solid rgba(0,0,0,0.06);flex-shrink:0;padding:0 4px}
.vrmi-tab{flex:1;padding:8px 2px;font-size:11px;font-weight:500;background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;color:#86868b;min-height:auto;margin:0;border-radius:0;transition:color 0.2s,border-color 0.2s}
.vrmi-tab:hover{color:#1d1d1f;background:none}
.vrmi-tab.active{color:#0071e3;border-bottom-color:#0071e3;background:none}
.vrmi-body{flex:1;overflow-y:auto;padding:12px 14px;font-size:13px;color:#424245}
.vrmi-tab-panel{display:none}
.vrmi-tab-panel.active{display:block}
.vrmi-reset{display:block;padding:5px 14px;font-size:12px;font-weight:500;color:#86868b;background:rgba(0,0,0,0.03);border:none;border-radius:6px;cursor:pointer;min-height:auto;flex-shrink:0;margin:8px 14px;transition:background 0.2s,color 0.2s}
.vrmi-reset:hover{color:#1d1d1f;background:rgba(0,0,0,0.07)}
.vrmi-sliders{display:grid;grid-template-columns:80px 1fr 42px;gap:4px 8px;align-items:center;margin-top:6px}
.vrmi-sliders input[type="range"]{-webkit-appearance:none;appearance:none;width:100%;height:4px;background:rgba(0,0,0,0.1);border-radius:2px;outline:none;cursor:pointer}
.vrmi-sliders input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;background:#fff;border-radius:50%;box-shadow:0 0.5px 2px rgba(0,0,0,0.2),0 0 0 0.5px rgba(0,0,0,0.08);cursor:pointer;transition:box-shadow 0.15s}
.vrmi-sliders input[type="range"]::-webkit-slider-thumb:hover{box-shadow:0 0.5px 4px rgba(0,0,0,0.25),0 0 0 0.5px rgba(0,0,0,0.12)}
.vrmi-sliders input[type="range"]::-moz-range-thumb{width:16px;height:16px;background:#fff;border:none;border-radius:50%;box-shadow:0 0.5px 2px rgba(0,0,0,0.2),0 0 0 0.5px rgba(0,0,0,0.08);cursor:pointer}
.vrmi-sliders input[type="range"]::-moz-range-track{height:4px;background:rgba(0,0,0,0.1);border-radius:2px;border:none}
.vrmi-switch{position:relative;width:42px;height:24px;-webkit-appearance:none;appearance:none;background:rgba(0,0,0,0.12);border-radius:12px;cursor:pointer;transition:background 0.25s;outline:none;border:none;flex-shrink:0}
.vrmi-switch::before{content:'';position:absolute;top:2px;left:2px;width:20px;height:20px;background:#fff;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,0.2);transition:transform 0.25s}
.vrmi-switch:checked{background:#34c759}
.vrmi-switch:checked::before{transform:translateX(18px)}
.vrmi-sway-target{margin:4px 0;border:1px solid rgba(0,0,0,0.08);border-radius:6px;background:rgba(255,255,255,0.35)}
.vrmi-sway-target>summary{display:flex;align-items:center;justify-content:space-between;padding:4px 8px;font-size:12px;font-weight:600;cursor:pointer;user-select:none;list-style:none;color:#1d1d1f}
.vrmi-sway-target>summary::-webkit-details-marker{display:none}
.vrmi-sway-target>summary>span:first-child::before{content:'\\25b6';display:inline-block;margin-right:5px;font-size:9px;transition:transform 0.2s}
.vrmi-sway-target[open]>summary>span:first-child::before{transform:rotate(90deg)}
.vrmi-sway-remove{all:unset;cursor:pointer;font-size:14px;color:#86868b;padding:0 4px;line-height:1;transition:color 0.2s}
.vrmi-sway-remove:hover{color:#ff3b30}
.vrmi-sway-target .vrmi-sliders{padding:2px 8px 6px}
.vrmi-sway-add{display:flex;gap:6px;align-items:center;margin-top:6px}
.vrmi-sway-add select{padding:3px 6px;font-size:12px;border:1px solid rgba(0,0,0,0.1);border-radius:6px;background:rgba(255,255,255,0.5)}
.vrmi-sampler-btn{padding:4px 10px;font-size:12px;font-weight:500;color:#424245;background:rgba(255,255,255,0.45);border:1px solid rgba(0,0,0,0.1);border-radius:6px;cursor:pointer;min-height:auto;margin:0;transition:background 0.15s,border-color 0.15s}
.vrmi-sampler-btn:hover{background:rgba(255,255,255,0.75);border-color:rgba(0,0,0,0.15)}
.vrmi-sampler-btn.active{background:#0071e3;color:#fff;border-color:#0071e3}
`;
        document.head.appendChild(style);
    }

    /**
     * Create an inspector overlay UI.
     * @param {HTMLElement} [container=document.body] - Element to append the inspector to.
     * @returns {{ addTab: Function, onReset: Function }} Inspector API.
     */
    createInspector(container = document.body) {
        VRMIdle._injectInspectorCSS();
        const self = this;

        // --- Storage ---
        const POSE_KEY = 'vrm_idle_pose';
        const SWAY_KEY = 'vrm_idle_sway';

        // --- Toggle button ---
        const toggle = document.createElement('button');
        toggle.className = 'vrmi-toggle';
        toggle.innerHTML = '\u2699';

        // --- Panel ---
        const panel = document.createElement('div');
        panel.className = 'vrmi-panel';

        // Header
        const header = document.createElement('div');
        header.className = 'vrmi-header';
        const title = document.createElement('span');
        title.textContent = 'Inspector';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'vrmi-close';
        closeBtn.innerHTML = '\u00d7';
        header.appendChild(title);
        header.appendChild(closeBtn);

        // Tabs bar
        const tabsBar = document.createElement('div');
        tabsBar.className = 'vrmi-tabs';

        // Body
        const body = document.createElement('div');
        body.className = 'vrmi-body';

        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'vrmi-reset';
        resetBtn.textContent = 'Reset settings';

        panel.appendChild(header);
        panel.appendChild(tabsBar);
        panel.appendChild(body);
        panel.appendChild(resetBtn);
        container.appendChild(toggle);
        container.appendChild(panel);

        // --- Toggle / close logic ---
        toggle.addEventListener('click', () => {
            panel.classList.add('open');
            toggle.style.display = 'none';
        });
        closeBtn.addEventListener('click', () => {
            panel.classList.remove('open');
            toggle.style.display = '';
        });

        // --- Tab management ---
        const tabs = [];
        const resetCallbacks = [];
        let activeTabIdx = 0;

        function switchTab(idx) {
            activeTabIdx = idx;
            tabs.forEach((t, i) => {
                t.btn.classList.toggle('active', i === idx);
                t.panel.classList.toggle('active', i === idx);
            });
            if (tabs[idx]) resetBtn.textContent = `Reset ${tabs[idx].label}`;
        }

        function addTab(label, buildFn, options = {}) {
            const pos = options.position ?? tabs.length;
            const btn = document.createElement('button');
            btn.className = 'vrmi-tab';
            btn.textContent = label;
            const tabPanel = document.createElement('div');
            tabPanel.className = 'vrmi-tab-panel';
            const entry = { btn, panel: tabPanel, buildFn, label, resetFn: null };
            tabs.splice(pos, 0, entry);

            // Rebuild DOM order
            tabsBar.innerHTML = '';
            body.innerHTML = '';
            tabs.forEach((t, i) => {
                tabsBar.appendChild(t.btn);
                body.appendChild(t.panel);
                t.btn.onclick = () => switchTab(i);
            });

            if (buildFn) buildFn(tabPanel);

            // Activate
            if (options.active) {
                switchTab(pos);
            } else if (tabs.length === 1) {
                switchTab(0);
            }

            return tabPanel;
        }

        // --- Build Pose tab ---
        addTab('Pose', (p) => this._buildPosePanel(p, POSE_KEY));

        // --- Build Sway tab ---
        addTab('Sway', (p) => this._buildSwayPanel(p, SWAY_KEY));

        // Activate first tab
        switchTab(0);

        // --- Per-tab reset ---
        resetBtn.addEventListener('click', () => {
            const tab = tabs[activeTabIdx];
            if (!tab) return;
            if (!confirm(`Reset "${tab.label}" settings to defaults?`)) return;

            // Built-in tab resets
            if (tab.label === 'Pose') {
                for (const pd of VRMIdle.POSE_PARAMS) {
                    self.pose[pd.key] = pd.def;
                }
                localStorage.removeItem(POSE_KEY);
            } else if (tab.label === 'Sway') {
                self.swayReset();
                localStorage.removeItem(SWAY_KEY);
            }

            // Per-tab external reset callback
            if (tab.resetFn) tab.resetFn();

            // Rebuild this tab's panel
            tab.panel.innerHTML = '';
            if (tab.buildFn) tab.buildFn(tab.panel);
        });

        return {
            addTab,
            onReset(fn) { resetCallbacks.push(fn); },
            onTabReset(label, fn) {
                const tab = tabs.find(t => t.label === label);
                if (tab) tab.resetFn = fn;
            },
            element: panel,
            toggle,
        };
    }

    // ================================================================
    // Inspector: Pose Panel
    // ================================================================
    _buildPosePanel(container, storageKey) {
        // Load saved values
        let saved = {};
        try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch {}
        for (const pd of VRMIdle.POSE_PARAMS) {
            this.pose[pd.key] = saved[pd.key] ?? pd.def;
        }

        const grid = document.createElement('div');
        grid.className = 'vrmi-sliders';
        for (const pd of VRMIdle.POSE_PARAMS) {
            const label = document.createElement('span');
            label.textContent = pd.label;
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = pd.min;
            slider.max = pd.max;
            slider.value = this.pose[pd.key];
            const display = document.createElement('span');
            display.textContent = pd.fmt(this.pose[pd.key]);
            slider.addEventListener('input', () => {
                this.pose[pd.key] = parseFloat(slider.value);
                display.textContent = pd.fmt(this.pose[pd.key]);
                const data = {};
                for (const p of VRMIdle.POSE_PARAMS) data[p.key] = this.pose[p.key];
                localStorage.setItem(storageKey, JSON.stringify(data));
            });
            grid.appendChild(label);
            grid.appendChild(slider);
            grid.appendChild(display);
        }
        container.appendChild(grid);
    }

    // ================================================================
    // Inspector: Sway Panel
    // ================================================================
    _buildSwayPanel(container, storageKey) {
        const self = this;

        // Enable/disable toggle
        const toggleRow = document.createElement('div');
        toggleRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:8px';
        const toggleLabel = document.createElement('span');
        toggleLabel.textContent = 'Enabled';
        toggleLabel.style.cssText = 'font-size:12px;font-weight:500;color:#424245';
        const toggleInput = document.createElement('input');
        toggleInput.type = 'checkbox';
        toggleInput.className = 'vrmi-switch';
        toggleInput.checked = this.swayEnabled;
        toggleInput.addEventListener('change', () => {
            const enabling = toggleInput.checked;
            self._swayAutoDisabled = false;
            self.setSwayEnabled(enabling, true);
        });
        toggleRow.appendChild(toggleLabel);
        toggleRow.appendChild(toggleInput);
        container.appendChild(toggleRow);

        // Load saved sway data
        let saved = null;
        try { saved = JSON.parse(localStorage.getItem(storageKey)); } catch {}
        this.swayLoadData(saved);

        function save() {
            localStorage.setItem(storageKey, JSON.stringify(self.swaySaveData()));
        }

        const targetsDiv = document.createElement('div');

        function render() {
            targetsDiv.innerHTML = '';
            self.swayTargets.forEach((target, idx) => {
                const details = document.createElement('details');
                details.className = 'vrmi-sway-target';
                const summary = document.createElement('summary');
                const nameSpan = document.createElement('span');
                nameSpan.textContent = target.bone;
                summary.appendChild(nameSpan);
                const removeBtn = document.createElement('button');
                removeBtn.className = 'vrmi-sway-remove';
                removeBtn.textContent = '\u00d7';
                removeBtn.addEventListener('click', (e) => {
                    e.preventDefault(); e.stopPropagation();
                    self.swayRemove(idx);
                    save();
                    render();
                });
                summary.appendChild(removeBtn);
                details.appendChild(summary);

                const grid = document.createElement('div');
                grid.className = 'vrmi-sliders';
                for (const pd of VRMIdle.SWAY_PARAMS) {
                    const label = document.createElement('span');
                    label.textContent = pd.label;
                    const slider = document.createElement('input');
                    slider.type = 'range';
                    slider.min = pd.min;
                    slider.max = pd.max;
                    slider.value = target.params[pd.key];
                    const display = document.createElement('span');
                    display.textContent = pd.fmt(target.params[pd.key]);
                    slider.addEventListener('input', () => {
                        target.params[pd.key] = parseFloat(slider.value);
                        display.textContent = pd.fmt(target.params[pd.key]);
                        save();
                    });
                    grid.appendChild(label);
                    grid.appendChild(slider);
                    grid.appendChild(display);
                }
                details.appendChild(grid);
                targetsDiv.appendChild(details);
            });
        }

        container.appendChild(targetsDiv);

        // Add bone row
        const addRow = document.createElement('div');
        addRow.className = 'vrmi-sway-add';
        const sel = document.createElement('select');
        for (const b of VRMIdle.SWAY_BONES) {
            const opt = document.createElement('option');
            opt.value = b; opt.textContent = b;
            sel.appendChild(opt);
        }
        const addBtn = document.createElement('button');
        addBtn.className = 'vrmi-sampler-btn';
        addBtn.textContent = 'Add';
        addBtn.addEventListener('click', () => {
            self.swayAdd(sel.value);
            save();
            render();
        });
        addRow.appendChild(sel);
        addRow.appendChild(addBtn);
        container.appendChild(addRow);

        render();
    }

}
