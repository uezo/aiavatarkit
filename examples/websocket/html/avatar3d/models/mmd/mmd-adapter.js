import { createMmdSettingsHost } from "./mmd-settings-host.js";
import { installMmdSettings } from "./mmd-settings.js";

const MODEL_FILE_PATTERN = /\.(pmx|pmd|bpmx)$/i;
const MOTION_FILE_PATTERN = /\.(vmd|bvmd)$/i;

function extensionFromName(name) {
    const lower = String(name || "").toLowerCase();
    if (lower.includes(".pmd")) return ".pmd";
    if (lower.includes(".bpmx")) return ".bpmx";
    return ".pmx";
}

function storedObject(key, defaults, enabled) {
    if (!enabled || !key) return { ...defaults };
    try {
        return { ...defaults, ...JSON.parse(localStorage.getItem(key) || "{}") };
    } catch {
        return { ...defaults };
    }
}

function storeObject(key, value, enabled) {
    if (enabled && key) localStorage.setItem(key, JSON.stringify(value));
}

export class MmdAdapter {
    constructor({ config, persistence, blobStore }) {
        this.config = config;
        this.persistence = persistence;
        this.blobStore = blobStore;
        this.B = globalThis.BABYLON;
        this.M = globalThis.BABYLONMMD;
        this.currentContainer = null;
        this.currentMesh = null;
        this.currentModel = null;
        this.currentMotion = null;
        this.runtime = null;
        this.physicsRuntime = null;
        this.pendingReferenceFiles = [];
        this.cameraReady = false;
        this.cameraSaveTimer = null;
        this.activeExpressionMorphs = new Set();
        this.expressionTimers = new Map();
        this.currentExpressionKey = "neutral";
        this.mouthTargets = { a: 0, u: 0, e: 0 };
        this.mouthValues = { a: 0, u: 0, e: 0 };
        this.blinkValue = 0;
        this.blinkPhase = "idle";
        this.idleBones = null;
        this.lastRenderTime = performance.now();
    }

    async initialize({ aiavatar, ui }) {
        if (!this.B || !this.M) throw new Error("Babylon.js and babylon-mmd must be loaded before MmdAdapter");
        this.aiavatar = aiavatar;
        this.ui = ui;
        this.canvas = document.getElementById("avatarCanvas");
        this.placeholder = document.getElementById("avatarPlaceholder");
        this.placeholderText = this.placeholder.textContent;
        this.loadSettings();
        this.settingsHost = createMmdSettingsHost();
        this.createScene();
        this.bindLoader();
        this.bindAvatar(aiavatar);
        installMmdSettings(this);
        this.startRenderLoop();
        await this.restoreAssets();
        return this.settingsHost;
    }

    loadSettings() {
        const restore = this.persistence.enabled && this.persistence.restoreUserSettings;
        this.physics = storedObject(this.persistence.physicsKey, this.config.physics, restore);
        const legacyPhysics = restore ? localStorage.getItem(this.persistence.legacyPhysicsEnabledKey) : null;
        if (legacyPhysics !== null) this.physics.enabled = legacyPhysics === "true";
        this.idle = storedObject(this.persistence.idleKey, this.config.idle, restore);
        this.appearance = storedObject(this.persistence.appearanceKey, this.config.appearance, restore);
        this.expressionMapping = storedObject(this.persistence.expressionMappingKey, {}, restore);
        this.faceMotion = storedObject(this.persistence.faceMotionKey, this.config.faceMotion, restore);
    }

    createScene() {
        const B = this.B;
        const engineConfig = this.config.engine;
        this.engine = new B.Engine(this.canvas, engineConfig.antialias, {
            preserveDrawingBuffer: engineConfig.preserveDrawingBuffer,
            stencil: engineConfig.stencil,
            alpha: engineConfig.alpha,
        });
        this.scene = new B.Scene(this.engine);
        this.scene.clearColor = new B.Color4(...engineConfig.clearColor);
        this.scene.ambientColor = new B.Color3(...this.config.lighting.sceneAmbientColor);

        const camera = this.config.camera;
        this.camera = new B.ArcRotateCamera(
            "camera",
            camera.alpha,
            camera.beta,
            camera.radius,
            new B.Vector3(...camera.target),
            this.scene,
        );
        this.camera.lowerRadiusLimit = camera.minRadius;
        this.camera.upperRadiusLimit = camera.maxRadius;
        this.camera.wheelPrecision = camera.wheelPrecision;
        this.camera.panningSensibility = camera.panningSensibility;
        this.camera.attachControl(this.canvas, true);
        this.cameraObserver = this.camera.onViewMatrixChangedObservable.add(() => this.scheduleCameraSave());

        this.hemiLight = new B.HemisphericLight("hemisphericLight", new B.Vector3(0, 1, 0), this.scene);
        this.hemiLight.groundColor = new B.Color3(...this.config.lighting.groundColor);
        this.dirLight = new B.DirectionalLight(
            "directionalLight",
            new B.Vector3(...this.config.lighting.direction),
            this.scene,
        );
        this.dirLight.position = new B.Vector3(...this.config.lighting.directionalPosition);
        this.faceLight = new B.PointLight(
            "faceLight",
            new B.Vector3(...this.config.lighting.facePosition),
            this.scene,
        );
        this.faceLight.range = this.config.lighting.faceRange;
        this.faceLight.diffuse = new B.Color3(...this.config.lighting.faceColor);
        this.applyAppearance();

        this.onResize = () => this.engine.resize();
        window.addEventListener("resize", this.onResize);
    }

    bindLoader() {
        this.M.SdefInjector?.OverrideEngineCreateEffect(this.engine);
        this.M.RegisterMmdModelLoaders?.();
        this.scene.getMmdOutlineRenderer?.();
        const observable = this.B.SceneLoader?.OnPluginActivatedObservable;
        if (!observable) return;
        this.loaderObserver = observable.add((loader) => {
            if (!["pmx", "pmd", "bpmx", "mmdmodel"].includes(loader.name)) return;
            loader.materialBuilder = new this.M.MmdStandardMaterialBuilder();
            loader.useSdef = true;
            loader.referenceFiles = this.pendingReferenceFiles;
            loader.loggingEnabled = this.config.loader.loggingEnabled;
        });
    }

    bindAvatar(aiavatar) {
        aiavatar.updateFace = (name, duration) => {
            this.applyExpression(name, duration ?? this.config.expression.defaultDurationSeconds);
        };
        aiavatar.resetFace = () => {
            this.applyExpression(this.config.expression.neutralName, 0);
            aiavatar.onResetFace?.();
        };
        this.lipsyncEngine = new LipSyncEngine(this.config.lipsync);
        aiavatar.onPlaybackAnalyze = ({ rms, centroid01, tSec }) => {
            const shape = this.lipsyncEngine.update({
                rms: rms * this.faceMotion.lipGain,
                centroid01,
                tSec,
            });
            this.applyViseme(shape);
        };
        aiavatar.onResetFace = () => this.clearVisemes();
        aiavatar.onPlaybackEnd = () => this.clearVisemes();
        this.configureBlinkController();
    }

    async ensureRuntime() {
        if (this.runtime) return this.runtime;
        try {
            const instance = await this.M.GetMmdWasmInstance(new this.M.MmdWasmInstanceTypeSPR());
            this.physicsRuntime = new this.M.PhysicsRuntime(instance);
            this.applyPhysicsGravity();
            this.physicsRuntime.register(this.scene);
            this.runtime = new this.M.MmdRuntime(
                this.scene,
                new this.M.MmdBulletPhysics(this.physicsRuntime),
            );
        } catch (error) {
            console.warn("MMD physics runtime unavailable; continuing without physics.", error);
            this.runtime = new this.M.MmdRuntime(this.scene, null);
        }
        this.runtime.register(this.scene);
        return this.runtime;
    }

    applyPhysicsGravity() {
        this.physicsRuntime?.setGravity?.(new this.B.Vector3(0, -Math.max(0, this.physics.gravity), 0));
    }

    relativePath(file) {
        return file.webkitRelativePath || file.name;
    }

    async prepareReferenceFiles(modelFile, files) {
        const modelPath = this.relativePath(modelFile);
        const slash = modelPath.lastIndexOf("/");
        const modelDirectory = slash >= 0 ? modelPath.slice(0, slash + 1) : "";
        const references = [];
        for (const file of files) {
            let relativePath = this.relativePath(file);
            if (modelDirectory && relativePath.startsWith(modelDirectory)) {
                relativePath = relativePath.slice(modelDirectory.length);
            }
            if (!relativePath || file === modelFile || MODEL_FILE_PATTERN.test(relativePath)) continue;
            references.push({
                relativePath,
                mimeType: file.type || undefined,
                data: await file.arrayBuffer(),
            });
        }
        return references;
    }

    async loadModelUrl(url, extension = extensionFromName(url)) {
        return this.loadModelSource(url, extension, []);
    }

    async loadModelFiles(modelFile, files = [modelFile], { cache = false } = {}) {
        const referenceFiles = await this.prepareReferenceFiles(modelFile, files);
        const source = new Uint8Array(await modelFile.arrayBuffer());
        const extension = extensionFromName(modelFile.name);
        const model = await this.loadModelSource(source, extension, referenceFiles);
        if (cache) {
            await Promise.all([
                this.blobStore.put(this.persistence.modelKey, modelFile),
                this.blobStore.put(this.persistence.modelExtensionKey, extension),
                this.blobStore.put(this.persistence.referenceFilesKey, referenceFiles),
            ]);
        }
        return model;
    }

    async loadModelSource(source, extension, referenceFiles) {
        await this.ensureRuntime();
        this.disposeCurrentModel();
        this.placeholder.style.display = "";
        this.placeholder.textContent = "Loading MMD model…";
        this.pendingReferenceFiles = referenceFiles;
        let container;
        try {
            container = await this.B.SceneLoader.LoadAssetContainerAsync(
                "",
                source,
                this.scene,
                undefined,
                extension,
            );
            container.addAllToScene();
            // MmdRuntime validates these two metadata fields. Prefer the
            // validated root instead of relying on container ordering because
            // legacy SceneLoader and different Babylon versions can prepend a
            // synthetic root mesh.
            const mesh = container.meshes.find((item) => (
                item.metadata?.isMmdModel === true
                && item.metadata?.skeleton != null
            )) || container.meshes[0];
            if (!mesh) throw new Error("No mesh found in MMD model");

            if (mesh.metadata?.isMmdModel !== true || mesh.metadata?.skeleton == null) {
                const candidates = container.meshes.map((item, index) => ({
                    index,
                    name: item.name,
                    isMmdModel: item.metadata?.isMmdModel === true,
                    hasMetadataSkeleton: item.metadata?.skeleton != null,
                    hasMeshSkeleton: item.skeleton != null,
                }));
                throw new Error(`No valid MMD root mesh found: ${JSON.stringify(candidates)}`);
            }

            this.currentContainer = container;
            this.currentMesh = mesh;
            this.normalizeMaterials(container);
            this.currentModel = this.runtime.createMmdModel(mesh, {
                materialProxyConstructor: this.M.MmdStandardMaterialProxy,
                buildPhysics: this.physics.enabled,
            });
            this.setupIdle();
            if (this.currentMotion) await this.applyCurrentMotion();
            this.frameCamera(mesh);
            this.placeholder.textContent = this.placeholderText;
            this.placeholder.style.display = "none";
            this.onMorphListChanged?.();
            console.log("MMD model loaded");
            return this.currentModel;
        } catch (error) {
            if (this.currentContainer === container) {
                this.disposeCurrentModel();
            } else if (container) {
                container.removeAllFromScene();
                container.dispose();
            }
            this.placeholder.textContent = this.placeholderText;
            this.placeholder.style.display = "";
            throw error;
        } finally {
            this.pendingReferenceFiles = [];
        }
    }

    async reloadCachedModel() {
        const modelBlob = await this.blobStore.get(this.persistence.modelKey);
        if (!modelBlob) return false;
        const extension = await this.blobStore.get(this.persistence.modelExtensionKey)
            || extensionFromName(modelBlob.name);
        const references = await this.blobStore.get(this.persistence.referenceFilesKey) || [];
        await this.loadModelSource(new Uint8Array(await modelBlob.arrayBuffer()), extension, references);
        return true;
    }

    async unloadModel({ clearCache = false } = {}) {
        this.disposeCurrentModel();
        if (!clearCache) return;
        await Promise.all([
            this.blobStore.delete(this.persistence.modelKey),
            this.blobStore.delete(this.persistence.modelExtensionKey),
            this.blobStore.delete(this.persistence.referenceFilesKey),
        ]);
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.cameraKey);
    }

    disposeCurrentModel() {
        this.cameraReady = false;
        clearTimeout(this.cameraSaveTimer);
        if (this.currentModel && this.runtime) {
            try {
                this.runtime.destroyMmdModel(this.currentModel);
            } catch (error) {
                console.warn("Could not destroy MMD runtime model:", error);
            }
        }
        this.currentModel = null;
        this.currentMesh = null;
        this.idleBones = null;
        if (this.currentContainer) {
            this.currentContainer.removeAllFromScene();
            this.currentContainer.dispose();
        }
        this.currentContainer = null;
        this.placeholder.textContent = this.placeholderText;
        this.placeholder.style.display = "";
    }

    async loadMotionBlob(name, blob, { cache = false } = {}) {
        await this.ensureRuntime();
        const url = URL.createObjectURL(blob);
        try {
            const loader = new this.M.VmdLoader(this.scene);
            this.currentMotion = await loader.loadAsync(name, url);
            if (this.currentModel) await this.applyCurrentMotion();
            if (cache) {
                await Promise.all([
                    this.blobStore.put(this.persistence.motionKey, blob),
                    this.blobStore.put(this.persistence.motionNameKey, name),
                ]);
            }
            return this.currentMotion;
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    async applyCurrentMotion() {
        if (!this.currentModel || !this.currentMotion) return;
        this.resetIdleBones();
        const animation = this.currentModel.createRuntimeAnimation(this.currentMotion);
        this.currentModel.setRuntimeAnimation(animation);
        if (this.physics.enabled) this.runtime.initializeMmdModelPhysics(this.currentModel);
    }

    playMotion() {
        if (this.currentMotion) this.runtime?.playAnimation();
    }

    pauseMotion() {
        this.runtime?.pauseAnimation();
    }

    async clearMotion() {
        this.pauseMotion();
        this.currentMotion = null;
        await Promise.all([
            this.blobStore.delete(this.persistence.motionKey),
            this.blobStore.delete(this.persistence.motionNameKey),
        ]);
    }

    async restoreAssets() {
        if (!this.persistence.restoreUserSettings) return;
        try {
            await this.reloadCachedModel();
            const motion = await this.blobStore.get(this.persistence.motionKey);
            if (motion) {
                const name = await this.blobStore.get(this.persistence.motionNameKey) || "motion";
                await this.loadMotionBlob(name, motion);
            }
        } catch (error) {
            console.warn("Could not restore cached MMD assets:", error);
        }
    }

    async importFiles(files) {
        const modelFile = files.find((file) => MODEL_FILE_PATTERN.test(file.name));
        if (modelFile) await this.loadModelFiles(modelFile, files, { cache: true });
        const motionFile = files.find((file) => MOTION_FILE_PATTERN.test(file.name));
        if (motionFile) {
            const name = motionFile.name.replace(MOTION_FILE_PATTERN, "") || "motion";
            await this.loadMotionBlob(name, motionFile, { cache: true });
        }
    }

    frameCamera(mesh) {
        mesh.computeWorldMatrix(true);
        const vectors = mesh.getHierarchyBoundingVectors(true);
        const center = vectors.min.add(vectors.max).scale(0.5);
        const height = Math.max(1, vectors.max.y - vectors.min.y);
        this.camera.setTarget(new this.B.Vector3(center.x, center.y + height * this.config.camera.targetHeightRatio, center.z));
        this.camera.radius = Math.max(this.config.camera.minFrameRadius, height * this.config.camera.frameDistanceRatio);
        this.camera.beta = this.config.camera.beta;
        this.outlineReferenceRadius = this.camera.radius;
        const restored = this.restoreCameraState();
        this.cameraReady = true;
        this.applyToonStyle();
        if (!restored) this.saveCameraState();
    }

    scheduleCameraSave() {
        if (!this.cameraReady || !this.persistence.enabled) return;
        clearTimeout(this.cameraSaveTimer);
        this.cameraSaveTimer = setTimeout(
            () => this.saveCameraState(),
            this.config.camera.saveDebounceMs,
        );
    }

    saveCameraState() {
        if (!this.cameraReady || !this.persistence.enabled) return;
        const target = this.camera.target;
        const position = this.camera.position;
        localStorage.setItem(this.persistence.cameraKey, JSON.stringify({
            px: position.x,
            py: position.y,
            pz: position.z,
            alpha: this.camera.alpha,
            beta: this.camera.beta,
            radius: this.camera.radius,
            tx: target.x,
            ty: target.y,
            tz: target.z,
        }));
    }

    restoreCameraState() {
        if (!this.persistence.enabled || !this.persistence.restoreUserSettings) return false;
        try {
            const state = JSON.parse(localStorage.getItem(this.persistence.cameraKey) || "null");
            if (!state || ![state.tx, state.ty, state.tz].every(Number.isFinite)) return false;
            this.camera.setTarget(new this.B.Vector3(state.tx, state.ty, state.tz));
            if ([state.px, state.py, state.pz].every(Number.isFinite)) {
                this.camera.setPosition(new this.B.Vector3(state.px, state.py, state.pz));
            } else if ([state.alpha, state.beta, state.radius].every(Number.isFinite)) {
                this.camera.alpha = state.alpha;
                this.camera.beta = state.beta;
                this.camera.radius = state.radius;
            }
            return true;
        } catch {
            return false;
        }
    }

    normalizeMaterials(container) {
        for (const material of container.materials || []) {
            if ("renderOutline" in material && !material._avatar3dOriginalOutline) {
                material._avatar3dOriginalOutline = {
                    enabled: Boolean(material.renderOutline),
                    width: material.outlineWidth || 0,
                    color: material.outlineColor?.clone?.() || new this.B.Color3(0, 0, 0),
                    alpha: material.outlineAlpha ?? 1,
                };
            }
            if ("ignoreDiffuseWhenToonTextureIsNull" in material) {
                material.ignoreDiffuseWhenToonTextureIsNull = false;
            }
            if ("applyAmbientColorToDiffuse" in material) material.applyAmbientColorToDiffuse = true;
            if (material.diffuseColor) {
                const color = material.diffuseColor;
                material.ambientColor = new this.B.Color3(
                    Math.min(1, color.r * 0.75 + 0.18),
                    Math.min(1, color.g * 0.75 + 0.16),
                    Math.min(1, color.b * 0.75 + 0.16),
                );
                material.emissiveColor = new this.B.Color3(color.r * 0.08, color.g * 0.08, color.b * 0.08);
            }
            if (material.specularColor) material.specularColor = material.specularColor.scale(0.12);
            material.specularPower = Math.min(material.specularPower || 8, 12);
        }
        this.applyToonStyle();
    }

    toonTexture() {
        if (this._toonTexture) return this._toonTexture;
        const colors = this.config.appearance.toonRamp;
        this._toonTexture = new this.B.DynamicTexture(
            "avatar3dToonRamp",
            { width: colors.length, height: 1 },
            this.scene,
            false,
            this.B.Texture.NEAREST_SAMPLINGMODE,
        );
        const context = this._toonTexture.getContext();
        colors.forEach((color, index) => {
            context.fillStyle = color;
            context.fillRect(index, 0, 1, 1);
        });
        this._toonTexture.update(false);
        this._toonTexture.wrapU = this.B.Texture.CLAMP_ADDRESSMODE;
        this._toonTexture.wrapV = this.B.Texture.CLAMP_ADDRESSMODE;
        return this._toonTexture;
    }

    applyToonStyle() {
        if (!this.currentContainer) return;
        const radius = Math.max(0.001, this.camera.radius || 1);
        const reference = Math.max(0.001, this.outlineReferenceRadius || radius);
        const cameraScale = reference / radius;
        for (const material of this.currentContainer.materials || []) {
            if ("toonTexture" in material) {
                material.toonTexture = this.appearance.toonEnabled ? this.toonTexture() : null;
            }
            const original = material._avatar3dOriginalOutline;
            if ("renderOutline" in material && original) {
                const enabled = this.appearance.outlineScale > 0.001 && original.enabled && original.width > 0;
                material.renderOutline = enabled;
                material.outlineWidth = enabled
                    ? original.width * this.appearance.outlineScale * cameraScale
                    : 0;
                material.outlineColor = original.color.clone?.() || original.color;
                material.outlineAlpha = enabled ? original.alpha : 0;
            }
        }
    }

    applyAppearance() {
        this.hemiLight.intensity = this.appearance.ambient;
        this.dirLight.intensity = this.appearance.directional;
        this.faceLight.intensity = this.appearance.face;
        this.applyToonStyle();
    }

    setAppearance(key, value) {
        this.appearance[key] = value;
        this.applyAppearance();
        storeObject(this.persistence.appearanceKey, this.appearance, this.persistence.enabled);
    }

    resetAppearance() {
        this.appearance = { ...this.config.appearance };
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.appearanceKey);
        this.applyAppearance();
    }

    async setPhysicsEnabled(enabled) {
        this.physics.enabled = enabled;
        this.savePhysics();
        if (!this.currentModel) return true;
        return this.reloadCachedModel();
    }

    setGravity(value) {
        this.physics.gravity = value;
        this.savePhysics();
        this.applyPhysicsGravity();
    }

    savePhysics() {
        storeObject(this.persistence.physicsKey, this.physics, this.persistence.enabled);
        if (this.persistence.enabled && this.persistence.legacyPhysicsEnabledKey) {
            localStorage.setItem(this.persistence.legacyPhysicsEnabledKey, String(this.physics.enabled));
        }
    }

    setIdle(key, value) {
        this.idle[key] = value;
        storeObject(this.persistence.idleKey, this.idle, this.persistence.enabled);
        if (!this.idle.enabled) this.resetIdleBones();
    }

    resetIdle() {
        this.resetIdleBones();
        this.idle = { ...this.config.idle };
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.idleKey);
    }

    findRuntimeBone(names) {
        if (!this.currentModel?.runtimeBones) return null;
        const candidates = names.map((name) => name.toLowerCase());
        return this.currentModel.runtimeBones.find((bone) => candidates.includes(bone.name.toLowerCase())) || null;
    }

    captureIdleBone(names) {
        const bone = this.findRuntimeBone(names);
        if (!bone?.linkedBone?.rotationQuaternion) return null;
        return {
            bone,
            base: bone.linkedBone.rotationQuaternion.clone(),
            temp: new this.B.Quaternion(),
        };
    }

    setupIdle() {
        this.idleStartTime = performance.now();
        this.idleBones = {
            upper: this.captureIdleBone(["上半身", "upper body", "upperbody", "spine"]),
            upper2: this.captureIdleBone(["上半身2", "upper body2", "upperbody2", "chest"]),
            neck: this.captureIdleBone(["首", "neck"]),
            head: this.captureIdleBone(["頭", "head"]),
            leftShoulder: this.captureIdleBone(["左肩", "left shoulder", "leftshoulder"]),
            rightShoulder: this.captureIdleBone(["右肩", "right shoulder", "rightshoulder"]),
            leftArm: this.captureIdleBone(["左腕", "left arm", "leftarm"]),
            rightArm: this.captureIdleBone(["右腕", "right arm", "rightarm"]),
            leftElbow: this.captureIdleBone(["左ひじ", "左肘", "left elbow", "leftelbow"]),
            rightElbow: this.captureIdleBone(["右ひじ", "右肘", "right elbow", "rightelbow"]),
        };
    }

    resetIdleBones() {
        for (const entry of Object.values(this.idleBones || {})) {
            if (entry) entry.bone.linkedBone.rotationQuaternion.copyFrom(entry.base);
        }
    }

    setIdleRotation(entry, yaw = 0, pitch = 0, roll = 0) {
        if (!entry) return;
        this.B.Quaternion.RotationYawPitchRollToRef(yaw, pitch, roll, entry.temp);
        entry.base.multiplyToRef(entry.temp, entry.bone.linkedBone.rotationQuaternion);
    }

    updateIdle() {
        if (!this.idle.enabled || !this.currentModel || !this.idleBones || this.currentMotion) return;
        const time = (performance.now() - this.idleStartTime) / 1000;
        const breath = this.idle.breath / 100;
        const sway = this.idle.sway / 100;
        const arm = this.idle.armRelax * Math.PI / 180;
        const breathe = Math.sin(time * 2.2) * 0.018 * breath;
        const breathe2 = Math.sin(time * 2.2 + 0.7) * 0.010 * breath;
        const swayYaw = Math.sin(time * 0.55) * 0.030 * sway;
        const swayRoll = Math.sin(time * 0.42 + 0.9) * 0.025 * sway;
        this.setIdleRotation(this.idleBones.upper, swayYaw * 0.35, breathe * 0.45, swayRoll * 0.35);
        this.setIdleRotation(this.idleBones.upper2, swayYaw * 0.55, breathe + breathe2, swayRoll);
        this.setIdleRotation(this.idleBones.neck, -swayYaw * 0.20, -breathe * 0.15, -swayRoll * 0.20);
        this.setIdleRotation(this.idleBones.head, -swayYaw * 0.35, -breathe * 0.12, -swayRoll * 0.25);
        this.setIdleRotation(this.idleBones.leftShoulder, 0, 0, arm * 0.18);
        this.setIdleRotation(this.idleBones.rightShoulder, 0, 0, -arm * 0.18);
        this.setIdleRotation(this.idleBones.leftArm, 0, 0, arm);
        this.setIdleRotation(this.idleBones.rightArm, 0, 0, -arm);
        this.setIdleRotation(this.idleBones.leftElbow, 0, 0, arm * 0.18);
        this.setIdleRotation(this.idleBones.rightElbow, 0, 0, -arm * 0.18);
    }

    getMorphNames() {
        const morphs = this.currentModel?.morph?.morphs;
        return Array.isArray(morphs) ? morphs.map((morph) => morph?.name).filter(Boolean) : [];
    }

    findMorph(candidates) {
        const names = this.getMorphNames();
        const exact = new Map(names.map((name) => [name.toLowerCase(), name]));
        for (const candidate of candidates || []) {
            const found = exact.get(String(candidate).toLowerCase());
            if (found) return found;
        }
        return "";
    }

    setMorph(name, value) {
        if (!this.currentModel?.morph || !name) return false;
        try {
            if (typeof this.currentModel.morph.setMorphWeight === "function") {
                this.currentModel.morph.setMorphWeight(name, value);
                return true;
            }
            if (typeof this.currentModel.morph.setMorphWeightFromName === "function") {
                this.currentModel.morph.setMorphWeightFromName(name, value);
                return true;
            }
        } catch {
            return false;
        }
        return false;
    }

    setFirstMorph(candidates, value) {
        const name = this.findMorph(candidates);
        return name ? this.setMorph(name, value) : false;
    }

    expressionMorph(key, originalName = "") {
        if (Object.prototype.hasOwnProperty.call(this.expressionMapping, key)) {
            return this.expressionMapping[key] || "";
        }
        return this.findMorph(this.config.expression.fallbacks[key] || [originalName]);
    }

    clearExpressionMorphs() {
        for (const timer of this.expressionTimers.values()) clearTimeout(timer);
        this.expressionTimers.clear();
        for (const name of this.activeExpressionMorphs) this.setMorph(name, 0);
        this.activeExpressionMorphs.clear();
    }

    applyExpression(faceName, duration = this.config.expression.defaultDurationSeconds) {
        this.clearExpressionMorphs();
        const requested = String(faceName || this.config.expression.neutralName).toLowerCase();
        const key = requested === "surprise" ? "surprised" : requested;
        this.currentExpressionKey = key;
        this.blinkController?.setState?.({ currentFace: key });
        if (key !== this.config.expression.neutralName) this.blinkPhase = "opening";
        const morphName = this.expressionMorph(key, faceName);
        if (morphName) {
            this.setMorph(morphName, 1);
            this.activeExpressionMorphs.add(morphName);
        }
        if (key === this.config.expression.neutralName || duration <= 0) return;
        const timerKey = morphName || `__${key}`;
        this.expressionTimers.set(timerKey, setTimeout(() => {
            if (morphName) {
                this.setMorph(morphName, 0);
                this.activeExpressionMorphs.delete(morphName);
            }
            this.expressionTimers.delete(timerKey);
            this.applyExpression(this.config.expression.neutralName, 0);
        }, duration * 1000));
    }

    setExpressionMapping(key, morphName) {
        if (morphName === undefined) delete this.expressionMapping[key];
        else this.expressionMapping[key] = morphName;
        storeObject(
            this.persistence.expressionMappingKey,
            this.expressionMapping,
            this.persistence.enabled,
        );
    }

    resetExpressionMapping() {
        this.expressionMapping = {};
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.expressionMappingKey);
        this.applyExpression(this.config.expression.neutralName, 0);
        this.onMorphListChanged?.();
    }

    setFaceMotion(key, value) {
        this.faceMotion[key] = value;
        storeObject(this.persistence.faceMotionKey, this.faceMotion, this.persistence.enabled);
        if (key === "blinkInterval") this.configureBlinkController();
    }

    configureBlinkController() {
        if (typeof BlinkController !== "function") return;
        const average = Math.max(1, Number(this.faceMotion.blinkInterval) || 4);
        const range = {
            min: Math.max(700, average * 650),
            max: Math.max(1200, average * 1450),
        };
        if (!this.blinkController) {
            this.blinkController = new BlinkController({
                autoStart: false,
                minIntervalMs: range.min,
                maxIntervalMs: range.max,
                blinkDurationMs: this.config.faceMotion.blinkDurationMs,
                shouldBlink: () => this.faceMotion.blinkEnabled
                    && this.currentExpressionKey === this.config.expression.neutralName
                    && Boolean(this.blinkMorph()),
                onBlinkStart: () => { this.blinkPhase = "closing"; },
                onBlinkEnd: () => { this.blinkPhase = "opening"; },
            });
            this.blinkController.start();
        } else {
            this.blinkController.minIntervalMs = range.min;
            this.blinkController.maxIntervalMs = range.max;
            this.blinkController.scheduleNext();
        }
    }

    blinkMorph() {
        return this.faceMotion.blinkMorph
            || this.findMorph(this.config.faceMotion.blinkMorphCandidates);
    }

    applyViseme(shape) {
        const target = this.config.faceMotion.mouthShapes[shape]
            || this.config.faceMotion.mouthShapes.closed;
        Object.assign(this.mouthTargets, target);
    }

    clearVisemes() {
        Object.assign(this.mouthTargets, { a: 0, u: 0, e: 0 });
    }

    updateMouth(dt) {
        const response = 0.08 + (100 - this.faceMotion.lipSmooth) / 160;
        const amount = Math.min(1, Math.max(0.03, response * dt * 60));
        for (const key of ["a", "u", "e"]) {
            this.mouthValues[key] += (this.mouthTargets[key] - this.mouthValues[key]) * amount;
            this.setFirstMorph(this.config.faceMotion.mouthMorphCandidates[key], this.mouthValues[key]);
        }
    }

    updateBlink(dt) {
        const morph = this.blinkMorph();
        if (!this.faceMotion.blinkEnabled || !morph || this.currentExpressionKey !== this.config.expression.neutralName) {
            this.blinkPhase = "opening";
        }
        if (this.blinkPhase === "closing") {
            this.blinkValue = Math.min(1, this.blinkValue + this.config.faceMotion.blinkCloseSpeed * dt);
            if (this.blinkValue >= 1) this.blinkPhase = "hold";
        } else if (this.blinkPhase === "opening") {
            this.blinkValue = Math.max(0, this.blinkValue - this.config.faceMotion.blinkOpenSpeed * dt);
            if (this.blinkValue <= 0) this.blinkPhase = "idle";
        } else if (this.blinkPhase === "idle") {
            this.blinkValue = 0;
        }
        if (morph) this.setMorph(morph, this.blinkValue);
    }

    startRenderLoop() {
        this.beforeAnimationsObserver = this.scene.onBeforeAnimationsObservable.add(() => this.updateIdle());
        this.engine.runRenderLoop(() => {
            const now = performance.now();
            const dt = Math.min(0.05, (now - this.lastRenderTime) / 1000);
            this.lastRenderTime = now;
            this.updateMouth(dt);
            this.updateBlink(dt);
            if (this.currentContainer && Math.abs((this.camera.radius || 0) - (this.lastOutlineRadius || 0)) > 0.02) {
                this.lastOutlineRadius = this.camera.radius || 0;
                this.applyToonStyle();
            }
            const forward = this.camera.getForwardRay(18).direction;
            this.faceLight.position = this.camera.position.add(forward.scale(this.config.lighting.faceFollowDistance));
            this.scene.render();
        });
    }

    handleResponse(response) {
        if (response.avatar_control_request?.animation_name && this.currentMotion) this.playMotion();
        if (response.type === "chunk" && response.metadata?.is_first_chunk) {
            this.aiavatar.updateFace(this.config.expression.neutralName, 0);
        }
    }

    stop() {
        this.clearVisemes();
        this.pauseMotion();
    }

    dispose() {
        clearTimeout(this.cameraSaveTimer);
        this.clearExpressionMorphs();
        this.blinkController?.stop();
        this.disposeCurrentModel();
        this.engine.stopRenderLoop();
        window.removeEventListener("resize", this.onResize);
        if (this.cameraObserver) this.camera.onViewMatrixChangedObservable.remove(this.cameraObserver);
        if (this.beforeAnimationsObserver) this.scene.onBeforeAnimationsObservable.remove(this.beforeAnimationsObserver);
        if (this.loaderObserver) this.B.SceneLoader.OnPluginActivatedObservable.remove(this.loaderObserver);
        try {
            this.runtime?.unregister?.(this.scene);
            this.physicsRuntime?.unregister?.(this.scene);
        } catch {
            // Runtime implementations differ; engine disposal still releases scene resources.
        }
        this._toonTexture?.dispose();
        this.settingsHost?.dispose?.();
        this.engine.dispose();
    }
}
