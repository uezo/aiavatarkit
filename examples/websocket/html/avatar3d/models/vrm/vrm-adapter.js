import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from "@pixiv/three-vrm-animation";
import { installVrmSettings } from "./vrm-settings.js";

const ARTIFACT_CAMERA_REFERENCE = {
    modelHeight: 1.670250301549527,
    targetOffset: [-0.10330158393900676, 0, -0.1170766868649855],
    cameraOffset: [1.100777579891017, 0.2236689191280165, 3.040772395500192],
};
const SKELETON_HEAD_PADDING_RATIO = 0.18;
const SKELETON_FOOT_PADDING_RATIO = 0.04;

function kelvinToRgb(kelvin) {
    const temperature = kelvin / 100;
    let red;
    let green;
    let blue;
    if (temperature <= 66) {
        red = 255;
        green = 99.4708025861 * Math.log(temperature) - 161.1195681661;
        blue = temperature <= 19 ? 0 : 138.5177312231 * Math.log(temperature - 10) - 305.0447927307;
    } else {
        red = 329.698727446 * Math.pow(temperature - 60, -0.1332047592);
        green = 288.1221695283 * Math.pow(temperature - 60, -0.0755148492);
        blue = 255;
    }
    return [red, green, blue].map((value) => Math.min(255, Math.max(0, value)) / 255);
}

export class VrmAdapter {
    constructor({ config, persistence, blobStore }) {
        this.config = config;
        this.persistence = persistence;
        this.blobStore = blobStore;
        this.currentModel = null;
        this.modelSkeletonFrame = null;
        this.modelDefaultCameraState = null;
        this.renderRequest = null;
        this.cameraSaveTimer = null;
        this.artifactMode = false;
        this.normalCameraState = null;
        this.normalMaxDistance = null;
        this.onAnimationListChanged = () => {};
        this.lighting = { ...config.lighting };
        this.lightDefinitions = [
            { key: "ambient", label: "Ambient", min: 0, max: 5, step: 0.1, format: (value) => value.toFixed(1) },
            { key: "directional", label: "Direct", min: 0, max: 5, step: 0.1, format: (value) => value.toFixed(1) },
            { key: "horizontalAngle", label: "H angle", min: -180, max: 180, format: (value) => `${value}°` },
            { key: "verticalAngle", label: "V angle", min: -90, max: 90, format: (value) => `${value}°` },
            { key: "colorTemperature", label: "Color temp", min: 2000, max: 10000, format: (value) => `${value}K` },
        ];
    }

    async initialize({ aiavatar, ui }) {
        this.aiavatar = aiavatar;
        this.ui = ui;
        this.canvas = document.getElementById("avatarCanvas");
        this.controlSurface = document.getElementById("avatarControlSurface");
        this.placeholder = document.getElementById("avatarPlaceholder");

        this.idle = new VRMIdle({ isAudioPlaying: () => aiavatar.isAudioPlaying });
        this.idle.swayPauseWhen = () => aiavatar.isAudioPlaying || ui.isServerProcessing;
        this.idle.swayResumeDelay = this.config.idle.swayResumeDelaySeconds;
        this.idle.setAnimationFactory(THREE.AnimationMixer, createVRMAnimationClip);
        this.settingsHost = this.idle.createInspector();

        this.createScene();
        this.loadLighting();
        this.applyLighting();
        installVrmSettings(this);
        await this.bind(aiavatar);
        this.installResizeHandler();
        this.start();
        await this.restoreAssets();
        return this.settingsHost;
    }

    createScene() {
        const cameraConfig = this.config.camera;
        this.scene = new THREE.Scene();
        this.viewCamera = new THREE.PerspectiveCamera(
            cameraConfig.fov,
            window.innerWidth / window.innerHeight,
            cameraConfig.near,
            cameraConfig.far,
        );
        this.viewCamera.position.set(...cameraConfig.position);
        this.viewCamera.lookAt(...cameraConfig.target);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: this.config.renderer.alpha,
            antialias: this.config.renderer.antialias,
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        const maxPixelRatio = this.config.renderer.maxPixelRatio || window.devicePixelRatio;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;

        this.ambientLight = new THREE.AmbientLight(0xffffff, this.lighting.ambient);
        this.directionalLight = new THREE.DirectionalLight(0xffffff, this.lighting.directional);
        this.scene.add(this.ambientLight, this.directionalLight);

        this.controls = this.createControls(
            new THREE.Vector3(...cameraConfig.target),
            cameraConfig.maxDistance,
        );

        this.clock = new THREE.Clock();
        this.loader = new GLTFLoader();
        this.loader.register((parser) => new VRMLoaderPlugin(parser));
        this.loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
    }

    createControls(target, maxDistance) {
        const controls = new OrbitControls(this.viewCamera, this.controlSurface);
        controls.target.copy(target);
        controls.enableDamping = this.config.camera.enableDamping;
        controls.dampingFactor = this.config.camera.dampingFactor;
        controls.enablePan = this.config.camera.enablePan;
        controls.minDistance = this.config.camera.minDistance;
        controls.maxDistance = maxDistance;
        controls.update();
        controls.addEventListener("change", () => {
            clearTimeout(this.cameraSaveTimer);
            this.cameraSaveTimer = setTimeout(() => this.saveCameraState(), this.config.camera.saveDebounceMs);
        });
        return controls;
    }

    getRawSkeletonBones() {
        const humanoid = this.currentModel?.humanoid;
        const hips = humanoid?.getRawBoneNode("hips");
        const head = humanoid?.getRawBoneNode("head");
        const feet = [
            humanoid?.getRawBoneNode("leftFoot"),
            humanoid?.getRawBoneNode("rightFoot"),
        ].filter(Boolean);
        return hips && head && feet.length > 0 ? { hips, head, feet } : null;
    }

    captureRawSkeletonFrame() {
        const skeleton = this.getRawSkeletonBones();
        if (!skeleton) return null;

        this.currentModel.scene.updateMatrixWorld(true);
        const hips = skeleton.hips.getWorldPosition(new THREE.Vector3());
        const head = skeleton.head.getWorldPosition(new THREE.Vector3());
        const feet = skeleton.feet.map((bone) => bone.getWorldPosition(new THREE.Vector3()));
        const feetY = Math.min(...feet.map(({ y }) => y));
        const skeletonHeight = head.y - feetY;
        if (![hips.x, hips.y, hips.z, head.y, feetY, skeletonHeight].every(Number.isFinite)) {
            return null;
        }
        if (skeletonHeight <= 0) return null;

        const top = head.y + skeletonHeight * SKELETON_HEAD_PADDING_RATIO;
        const bottom = feetY - skeletonHeight * SKELETON_FOOT_PADDING_RATIO;
        const center = hips.clone();
        center.y = (top + bottom) / 2;
        return { center, height: top - bottom };
    }

    captureDefaultModelCameraState() {
        const head = this.getRawSkeletonBones()?.head;
        if (!head) return null;

        this.currentModel.scene.updateMatrixWorld(true);
        const headPosition = head.getWorldPosition(new THREE.Vector3());
        if (![headPosition.x, headPosition.y, headPosition.z].every(Number.isFinite)) return null;

        return {
            px: 0,
            py: headPosition.y,
            pz: this.config.camera.autoFrameDistance,
            tx: 0,
            ty: headPosition.y - 0.05,
            tz: 0,
        };
    }

    applyDefaultModelCamera({ resetControls = false } = {}) {
        const state = this.modelDefaultCameraState || this.captureDefaultModelCameraState();
        if (!state) return false;
        return this.applyCameraState(state, {
            resetControls,
            maxDistance: this.config.camera.maxDistance,
        });
    }

    updateControlSurface() {
        const skeleton = this.getRawSkeletonBones();
        if (!skeleton) {
            this.controlSurface.hidden = true;
            return;
        }

        this.viewCamera.updateMatrixWorld(true);
        const point = new THREE.Vector3();
        const projectBone = (bone) => {
            bone.getWorldPosition(point);
            point.project(this.viewCamera);
            if (![point.x, point.y, point.z].every(Number.isFinite)) return null;
            if (point.z < -1 || point.z > 1) return null;
            return {
                x: (point.x + 1) * window.innerWidth / 2,
                y: (1 - point.y) * window.innerHeight / 2,
            };
        };
        const projectedHips = projectBone(skeleton.hips);
        const projectedHead = projectBone(skeleton.head);
        const projectedFeet = skeleton.feet.map(projectBone).filter(Boolean);
        if (!projectedHips || !projectedHead || projectedFeet.length === 0) {
            this.controlSurface.hidden = true;
            return;
        }

        const feetY = Math.max(...projectedFeet.map(({ y }) => y));
        const skeletonHeight = feetY - projectedHead.y;
        if (!Number.isFinite(skeletonHeight) || skeletonHeight < 8) {
            this.controlSurface.hidden = true;
            return;
        }

        const topPadding = skeletonHeight * SKELETON_HEAD_PADDING_RATIO;
        const bottomPadding = skeletonHeight * SKELETON_FOOT_PADDING_RATIO;
        const projectedHeight = skeletonHeight + topPadding + bottomPadding;
        const controlWidth = Math.min(320, Math.max(140, projectedHeight * 0.22));
        let left = projectedHips.x - controlWidth / 2;
        let top = projectedHead.y - topPadding;
        let right = projectedHips.x + controlWidth / 2;
        let bottom = feetY + bottomPadding;
        if (right <= 0 || left >= window.innerWidth || bottom <= 0 || top >= window.innerHeight) {
            this.controlSurface.hidden = true;
            return;
        }

        left = Math.max(0, left);
        top = Math.max(0, top);
        right = Math.min(window.innerWidth, right);
        bottom = Math.min(window.innerHeight, bottom);
        if (right <= left || bottom <= top) {
            this.controlSurface.hidden = true;
            return;
        }

        const insets = [
            top,
            window.innerWidth - right,
            window.innerHeight - bottom,
            left,
        ].map(Math.round);
        const clipPath = `inset(${insets.map((value) => `${value}px`).join(" ")})`;
        if (this.controlSurface.style.clipPath !== clipPath) {
            this.controlSurface.style.clipPath = clipPath;
        }
        this.controlSurface.hidden = false;
    }

    async bind(aiavatar) {
        aiavatar.updateFace = (faceName, faceDuration) => {
            this.idle.applyExpression(faceName, faceDuration ?? this.config.expression.defaultDurationSeconds);
        };
        aiavatar.resetFace = () => {
            this.idle.applyExpression(this.config.expression.neutralName);
            aiavatar.onResetFace?.();
        };

        this.lipsyncEngine = this.config.lipsync?.engine;
        await this.lipsyncEngine.initialize();
        aiavatar.onPlaybackAudio = (audio) => {
            const result = this.lipsyncEngine.processAudioData(audio);
            this.idle.applyVisemeWeights(this.lipSyncWeights(result));
        };
        aiavatar.onResetFace = () => this.idle.clearVisemes();
        aiavatar.onPlaybackEnd = () => this.idle.clearVisemes();
    }

    lipSyncWeights(result) {
        const configuredMax = Number(this.config.lipsync.maxVisemeWeight ?? 1);
        const maxWeight = Number.isFinite(configuredMax)
            ? Math.min(1, Math.max(0, configuredMax))
            : 1;
        const scale = (weight) => Math.min(1, Math.max(0, Number(weight) || 0)) * maxWeight;
        const weights = this.config.lipsync.usePhonemeBlend
            ? result.visemes
            : { A: 0, I: 0, U: 0, E: 0, O: 0 };
        if (!this.config.lipsync.usePhonemeBlend && result.mainViseme in weights) {
            weights[result.mainViseme] = result.mainVisemeWeight;
        }
        return Object.fromEntries(
            Object.entries(weights).map(([viseme, weight]) => [viseme, scale(weight)]),
        );
    }

    async loadModelUrl(url, { cache = false } = {}) {
        if (cache) {
            try {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await this.loadModelBlob(await response.blob(), { cache: true });
            } catch (error) {
                console.warn("Could not cache remote VRM; loading URL directly:", error);
            }
        }
        const gltf = await this.loader.loadAsync(url);
        return this.useLoadedModel(gltf);
    }

    async loadModelBlob(blob, { cache = false } = {}) {
        const objectUrl = URL.createObjectURL(blob);
        try {
            const gltf = await this.loader.loadAsync(objectUrl);
            const model = await this.useLoadedModel(gltf);
            if (cache) await this.blobStore.put(this.persistence.modelKey, blob);
            return model;
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    async useLoadedModel(gltf) {
        const model = gltf.userData.vrm;
        if (!model) throw new Error("No VRM data found");
        this.disposeCurrentModel();
        this.currentModel = model;
        this.idle.vrm = model;
        VRMUtils.rotateVRM0(model);
        this.scene.add(model.scene);
        this.modelSkeletonFrame = this.captureRawSkeletonFrame();
        this.modelDefaultCameraState = this.captureDefaultModelCameraState();
        this.applyDefaultModelCamera();
        if (model.lookAt) model.lookAt.target = this.viewCamera;
        this.placeholder.style.display = "none";
        console.log("VRM loaded");
        return model;
    }

    disposeCurrentModel() {
        if (!this.currentModel) return;
        this.scene.remove(this.currentModel.scene);
        VRMUtils.deepDispose(this.currentModel.scene);
        this.currentModel = null;
        this.modelSkeletonFrame = null;
        this.modelDefaultCameraState = null;
        this.idle.vrm = null;
        this.controlSurface.hidden = true;
    }

    async unloadModel({ clearCache = false } = {}) {
        this.disposeCurrentModel();
        this.placeholder.style.display = "";
        if (clearCache) await this.clearModelCache();
    }

    async clearModelCache() {
        await this.blobStore.delete(this.persistence.modelKey);
        if (this.persistence.enabled) {
            localStorage.removeItem(this.persistence.cameraKey);
            localStorage.removeItem(this.artifactCameraKey);
        }
    }

    async loadAnimationBlob(name, blob, { cache = false } = {}) {
        const normalizedName = name.toLowerCase();
        const objectUrl = URL.createObjectURL(blob);
        try {
            const gltf = await this.loader.loadAsync(objectUrl);
            const animation = gltf.userData.vrmAnimations?.[0];
            if (!animation) throw new Error("No VRMAnimation data found");
            this.idle.registerVRMA(normalizedName, animation);
            if (cache) {
                await this.blobStore.put(this.animationKey(normalizedName), blob);
                const names = await this.cachedAnimationNames();
                if (!names.includes(normalizedName)) {
                    names.push(normalizedName);
                    await this.storeAnimationNames(names);
                }
            }
            return animation;
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    get animationNames() {
        return this.idle.vrmaNames;
    }

    animationKey(name) {
        return `${this.persistence.animationKeyPrefix}${name}`;
    }

    async cachedAnimationNames() {
        const names = await this.blobStore.get(this.persistence.animationNamesKey);
        return Array.isArray(names) ? names : [];
    }

    storeAnimationNames(names) {
        return this.blobStore.put(this.persistence.animationNamesKey, names);
    }

    async renameAnimation(name, nextName) {
        if (!this.idle.renameVRMA(name, nextName)) return false;
        const blob = await this.blobStore.get(this.animationKey(name));
        if (blob) {
            await this.blobStore.put(this.animationKey(nextName), blob);
            await this.blobStore.delete(this.animationKey(name));
        }
        const names = await this.cachedAnimationNames();
        const index = names.indexOf(name);
        if (index >= 0) names[index] = nextName;
        else names.push(nextName);
        await this.storeAnimationNames([...new Set(names)]);
        return true;
    }

    async removeAnimation(name) {
        this.idle.unregisterVRMA(name);
        await this.blobStore.delete(this.animationKey(name));
        await this.storeAnimationNames((await this.cachedAnimationNames()).filter((item) => item !== name));
    }

    async restoreAssets() {
        if (!this.persistence.restoreUserSettings) return;
        try {
            const modelBlob = await this.blobStore.get(this.persistence.modelKey);
            if (modelBlob) {
                await this.loadModelBlob(modelBlob);
                this.restoreCameraState();
            }
        } catch (error) {
            console.warn("Could not restore cached VRM:", error);
        }

        try {
            const originalNames = await this.cachedAnimationNames();
            const names = [...new Set(originalNames.map((name) => name.toLowerCase()))];
            for (const name of names) {
                let blob = await this.blobStore.get(this.animationKey(name));
                if (!blob) {
                    const originalName = originalNames.find((item) => item.toLowerCase() === name);
                    if (originalName) {
                        blob = await this.blobStore.get(this.animationKey(originalName));
                        if (blob && originalName !== name) {
                            await this.blobStore.put(this.animationKey(name), blob);
                            await this.blobStore.delete(this.animationKey(originalName));
                        }
                    }
                }
                if (blob) await this.loadAnimationBlob(name, blob);
            }
            if (names.join("\0") !== originalNames.join("\0")) await this.storeAnimationNames(names);
            this.onAnimationListChanged();
        } catch (error) {
            console.warn("Could not restore cached VRMA animations:", error);
        }
    }

    async importFiles(files) {
        for (const file of files) {
            const lowerName = file.name.toLowerCase();
            if (lowerName.endsWith(".vrm")) await this.loadModelBlob(file, { cache: true });
            if (lowerName.endsWith(".vrma")) {
                await this.loadAnimationBlob(file.name.replace(/\.vrma$/i, ""), file, { cache: true });
            }
        }
        this.onAnimationListChanged();
    }

    loadLighting() {
        if (!this.persistence.enabled || !this.persistence.restoreUserSettings) return;
        try {
            const saved = JSON.parse(localStorage.getItem(this.persistence.lightingKey) || "{}");
            this.lighting.ambient = saved.ambient > 5 ? saved.ambient / 100 : saved.ambient ?? this.lighting.ambient;
            this.lighting.directional = saved.direct > 5 ? saved.direct / 100 : saved.direct ?? saved.directional ?? this.lighting.directional;
            this.lighting.horizontalAngle = saved.hAngle ?? saved.horizontalAngle ?? this.lighting.horizontalAngle;
            this.lighting.verticalAngle = saved.vAngle ?? saved.verticalAngle ?? this.lighting.verticalAngle;
            this.lighting.colorTemperature = saved.temp ?? saved.colorTemperature ?? this.lighting.colorTemperature;
        } catch (error) {
            console.warn("Could not restore lighting settings:", error);
        }
    }

    saveLighting() {
        if (!this.persistence.enabled) return;
        localStorage.setItem(this.persistence.lightingKey, JSON.stringify({
            ambient: this.lighting.ambient,
            direct: this.lighting.directional,
            hAngle: this.lighting.horizontalAngle,
            vAngle: this.lighting.verticalAngle,
            temp: this.lighting.colorTemperature,
        }));
    }

    setLighting(key, value) {
        this.lighting[key] = value;
        this.applyLighting();
        this.saveLighting();
    }

    resetLighting() {
        this.lighting = { ...this.config.lighting };
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.lightingKey);
        this.applyLighting();
    }

    applyLighting() {
        this.ambientLight.intensity = this.lighting.ambient;
        this.directionalLight.intensity = this.lighting.directional;
        const horizontal = THREE.MathUtils.degToRad(this.lighting.horizontalAngle);
        const vertical = THREE.MathUtils.degToRad(this.lighting.verticalAngle);
        this.directionalLight.position.set(
            Math.cos(vertical) * Math.sin(horizontal),
            Math.sin(vertical),
            Math.cos(vertical) * Math.cos(horizontal),
        );
        this.directionalLight.color.setRGB(...kelvinToRgb(this.lighting.colorTemperature));
    }

    get artifactCameraKey() {
        return `${this.persistence.cameraKey}_artifact`;
    }

    captureCameraState() {
        return {
            px: this.viewCamera.position.x,
            py: this.viewCamera.position.y,
            pz: this.viewCamera.position.z,
            tx: this.controls.target.x,
            ty: this.controls.target.y,
            tz: this.controls.target.z,
        };
    }

    applyCameraState(state, { resetControls = false, maxDistance = this.controls.maxDistance } = {}) {
        const values = [state?.px, state?.py, state?.pz, state?.tx, state?.ty, state?.tz];
        if (!values.every(Number.isFinite)) return false;
        this.viewCamera.position.set(state.px, state.py, state.pz);
        const target = new THREE.Vector3(state.tx, state.ty, state.tz);
        if (resetControls) {
            this.controls.dispose();
            this.controls = this.createControls(target, maxDistance);
        } else {
            this.controls.target.copy(target);
            this.controls.update();
        }
        return true;
    }

    saveCameraState(key = this.artifactMode ? this.artifactCameraKey : this.persistence.cameraKey) {
        if (!this.persistence.enabled) return;
        const state = this.captureCameraState();
        if (key === this.artifactCameraKey) state.layout = "viewport";
        localStorage.setItem(key, JSON.stringify(state));
    }

    restoreCameraState(
        key = this.artifactMode ? this.artifactCameraKey : this.persistence.cameraKey,
        options = {},
    ) {
        if (!this.persistence.enabled || !this.persistence.restoreUserSettings) return false;
        try {
            const state = JSON.parse(localStorage.getItem(key));
            if (key === this.artifactCameraKey && state?.layout !== "viewport") return false;
            return this.applyCameraState(state, options);
        } catch {
            return false;
        }
    }

    applyArtifactViewOffset() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        this.viewCamera.setViewOffset(
            width,
            height,
            -0.36 * width,
            -0.26 * height,
            width,
            height,
        );
    }

    applyDefaultArtifactCamera(maxDistance) {
        const frame = this.modelSkeletonFrame || this.captureRawSkeletonFrame();
        if (!frame) return false;

        const scale = frame.height / ARTIFACT_CAMERA_REFERENCE.modelHeight;
        const target = frame.center.clone().add(
            new THREE.Vector3(...ARTIFACT_CAMERA_REFERENCE.targetOffset).multiplyScalar(scale),
        );
        const cameraOffset = new THREE.Vector3(...ARTIFACT_CAMERA_REFERENCE.cameraOffset)
            .multiplyScalar(scale * 2);
        const position = target.clone().add(cameraOffset);
        const distance = cameraOffset.length();
        return this.applyCameraState({
            px: position.x,
            py: position.y,
            pz: position.z,
            tx: target.x,
            ty: target.y,
            tz: target.z,
        }, { resetControls: true, maxDistance: Math.max(maxDistance, distance * 1.2) });
    }

    resetView() {
        if (!this.currentModel) return false;
        clearTimeout(this.cameraSaveTimer);
        this.cameraSaveTimer = null;
        this.viewCamera.zoom = 1;
        this.viewCamera.updateProjectionMatrix();

        let reset;
        let persistenceKey;
        if (this.artifactMode) {
            const normalMaxDistance = this.normalMaxDistance ?? this.config.camera.maxDistance;
            const maxDistance = Math.max(normalMaxDistance, this.config.camera.maxDistance * 3);
            reset = this.applyDefaultArtifactCamera(maxDistance);
            persistenceKey = this.artifactCameraKey;
        } else {
            reset = this.applyDefaultModelCamera({ resetControls: true });
            persistenceKey = this.persistence.cameraKey;
        }
        if (!reset) return false;

        this.saveCameraState(persistenceKey);
        this.controlSurface.hidden = true;
        return true;
    }

    setArtifactMode(active) {
        active = Boolean(active);
        if (active === this.artifactMode) return;
        clearTimeout(this.cameraSaveTimer);

        if (active) {
            this.normalCameraState = this.captureCameraState();
            this.normalMaxDistance = this.controls.maxDistance;
            this.saveCameraState(this.persistence.cameraKey);
            this.artifactMode = true;
            this.applyArtifactViewOffset();
            const maxDistance = Math.max(this.normalMaxDistance, this.config.camera.maxDistance * 3);
            if (!this.restoreCameraState(undefined, { resetControls: true, maxDistance })) {
                this.applyDefaultArtifactCamera(maxDistance);
            }
            this.controlSurface.hidden = true;
            return;
        }

        this.saveCameraState(this.artifactCameraKey);
        this.artifactMode = false;
        this.viewCamera.clearViewOffset();
        const maxDistance = this.normalMaxDistance ?? this.config.camera.maxDistance;
        if (!this.applyCameraState(this.normalCameraState, { resetControls: true, maxDistance })) {
            this.restoreCameraState(undefined, { resetControls: true, maxDistance });
        }
        this.normalCameraState = null;
        this.normalMaxDistance = null;
        this.controlSurface.hidden = true;
    }

    handleResponse(response) {
        const animationRequest = response.avatar_control_request;
        if (animationRequest?.animation_name) {
            this.idle.playAnimation(animationRequest.animation_name, animationRequest.animation_duration);
        }
        if (response.type === "chunk" && response.metadata?.is_first_chunk) {
            this.aiavatar.updateFace(this.config.expression.neutralName, 0);
        }
    }

    installResizeHandler() {
        this.onResize = () => {
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.viewCamera.aspect = window.innerWidth / window.innerHeight;
            if (this.artifactMode) this.applyArtifactViewOffset();
            else this.viewCamera.updateProjectionMatrix();
        };
        window.addEventListener("resize", this.onResize);
    }

    start() {
        if (this.renderRequest) return;
        const render = () => {
            this.renderRequest = requestAnimationFrame(render);
            this.controls.update();
            this.idle.update(this.clock.getDelta());
            this.updateControlSurface();
            this.renderer.render(this.scene, this.viewCamera);
        };
        render();
    }

    stop() {
        this.idle.clearVisemes();
    }

    dispose() {
        if (this.renderRequest) cancelAnimationFrame(this.renderRequest);
        this.renderRequest = null;
        clearTimeout(this.cameraSaveTimer);
        window.removeEventListener("resize", this.onResize);
        this.disposeCurrentModel();
        this.controls.dispose();
        this.renderer.dispose();
    }
}
