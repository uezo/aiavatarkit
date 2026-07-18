import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from "@pixiv/three-vrm-animation";
import { installVrmSettings } from "./vrm-settings.js";

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
        this.renderRequest = null;
        this.cameraSaveTimer = null;
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
        this.bind(aiavatar);
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

        this.controls = new OrbitControls(this.viewCamera, this.canvas);
        this.controls.target.set(...cameraConfig.target);
        this.controls.enableDamping = cameraConfig.enableDamping;
        this.controls.dampingFactor = cameraConfig.dampingFactor;
        this.controls.enablePan = cameraConfig.enablePan;
        this.controls.minDistance = cameraConfig.minDistance;
        this.controls.maxDistance = cameraConfig.maxDistance;
        this.controls.update();
        this.controls.addEventListener("change", () => {
            clearTimeout(this.cameraSaveTimer);
            this.cameraSaveTimer = setTimeout(() => this.saveCameraState(), this.config.camera.saveDebounceMs);
        });

        this.clock = new THREE.Clock();
        this.loader = new GLTFLoader();
        this.loader.register((parser) => new VRMLoaderPlugin(parser));
        this.loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
    }

    bind(aiavatar) {
        aiavatar.updateFace = (faceName, faceDuration) => {
            this.idle.applyExpression(faceName, faceDuration || this.config.expression.defaultDurationSeconds);
        };
        aiavatar.resetFace = () => {
            this.idle.applyExpression(this.config.expression.neutralName);
            aiavatar.onResetFace?.();
        };

        this.lipsyncEngine = new LipSyncEngine(this.config.lipsync);
        aiavatar.onPlaybackAnalyze = ({ rms, centroid01, tSec }) => {
            const shape = this.lipsyncEngine.update({ rms, centroid01, tSec });
            this.idle.applyViseme(shape);
        };
        aiavatar.onResetFace = () => this.idle.clearVisemes();
        aiavatar.onPlaybackEnd = () => this.idle.clearVisemes();
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

        const head = model.humanoid?.getNormalizedBoneNode("head");
        if (head) {
            const headPosition = new THREE.Vector3();
            head.getWorldPosition(headPosition);
            const distance = this.config.camera.autoFrameDistance;
            this.viewCamera.position.set(0, headPosition.y, distance);
            this.viewCamera.lookAt(0, headPosition.y - 0.05, 0);
            this.controls.target.set(0, headPosition.y - 0.05, 0);
            this.controls.update();
        }
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
        this.idle.vrm = null;
    }

    async unloadModel({ clearCache = false } = {}) {
        this.disposeCurrentModel();
        this.placeholder.style.display = "";
        if (clearCache) await this.clearModelCache();
    }

    async clearModelCache() {
        await this.blobStore.delete(this.persistence.modelKey);
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.cameraKey);
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

    saveCameraState() {
        if (!this.persistence.enabled) return;
        localStorage.setItem(this.persistence.cameraKey, JSON.stringify({
            px: this.viewCamera.position.x,
            py: this.viewCamera.position.y,
            pz: this.viewCamera.position.z,
            tx: this.controls.target.x,
            ty: this.controls.target.y,
            tz: this.controls.target.z,
        }));
    }

    restoreCameraState() {
        if (!this.persistence.enabled || !this.persistence.restoreUserSettings) return false;
        try {
            const saved = JSON.parse(localStorage.getItem(this.persistence.cameraKey));
            if (!saved) return false;
            this.viewCamera.position.set(saved.px, saved.py, saved.pz);
            this.controls.target.set(saved.tx, saved.ty, saved.tz);
            this.controls.update();
            return true;
        } catch {
            return false;
        }
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
            this.viewCamera.updateProjectionMatrix();
        };
        window.addEventListener("resize", this.onResize);
    }

    start() {
        if (this.renderRequest) return;
        const render = () => {
            this.renderRequest = requestAnimationFrame(render);
            this.controls.update();
            this.idle.update(this.clock.getDelta());
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
