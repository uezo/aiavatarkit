const ENTRY_STORE = "entries";
const META_STORE = "meta";
const CONTEXT_KEY = "contextId";

function requestResult(request) {
    return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

function transactionComplete(transaction) {
    return new Promise((resolve, reject) => {
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
    });
}

export function createBacklogStore({
    enabled = true,
    databaseName,
    maxEntries = 100,
}) {
    if (!databaseName) throw new Error("backlog databaseName is required");

    const entryLimit = Number.isInteger(maxEntries) && maxEntries > 0 ? maxEntries : 100;

    function open() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(databaseName, 1);
            request.onupgradeneeded = () => {
                const database = request.result;
                if (!database.objectStoreNames.contains(ENTRY_STORE)) {
                    const entries = database.createObjectStore(ENTRY_STORE, { keyPath: "id" });
                    entries.createIndex("createdAt", "createdAt");
                }
                if (!database.objectStoreNames.contains(META_STORE)) {
                    database.createObjectStore(META_STORE);
                }
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async function load() {
        if (!enabled) return { contextId: null, entries: [] };
        const database = await open();
        try {
            const transaction = database.transaction([ENTRY_STORE, META_STORE], "readonly");
            const done = transactionComplete(transaction);
            const contextRequest = transaction.objectStore(META_STORE).get(CONTEXT_KEY);
            const entriesRequest = transaction.objectStore(ENTRY_STORE).index("createdAt").getAll();
            const [contextId, entries] = await Promise.all([
                requestResult(contextRequest),
                requestResult(entriesRequest),
            ]);
            await done;
            return { contextId: contextId || null, entries: entries || [] };
        } finally {
            database.close();
        }
    }

    async function clear() {
        if (!enabled) return;
        const database = await open();
        try {
            const transaction = database.transaction([ENTRY_STORE, META_STORE], "readwrite");
            const done = transactionComplete(transaction);
            transaction.objectStore(ENTRY_STORE).clear();
            transaction.objectStore(META_STORE).delete(CONTEXT_KEY);
            await done;
        } finally {
            database.close();
        }
    }

    async function trim() {
        if (!enabled) return;
        const database = await open();
        try {
            const transaction = database.transaction(ENTRY_STORE, "readwrite");
            const done = transactionComplete(transaction);
            const entries = transaction.objectStore(ENTRY_STORE);
            const count = await requestResult(entries.count());
            let remaining = Math.max(0, count - entryLimit);
            if (remaining > 0) {
                await new Promise((resolve, reject) => {
                    const request = entries.index("createdAt").openCursor();
                    request.onerror = () => reject(request.error);
                    request.onsuccess = () => {
                        const cursor = request.result;
                        if (!cursor || remaining <= 0) {
                            resolve();
                            return;
                        }
                        cursor.delete();
                        remaining -= 1;
                        cursor.continue();
                    };
                });
            }
            await done;
        } finally {
            database.close();
        }
    }

    async function appendTurn(contextId, entries) {
        if (!enabled || !contextId || !entries.length) return;
        const database = await open();
        try {
            const transaction = database.transaction([ENTRY_STORE, META_STORE], "readwrite");
            const done = transactionComplete(transaction);
            const entryStore = transaction.objectStore(ENTRY_STORE);
            const metaStore = transaction.objectStore(META_STORE);
            const storedContextId = await requestResult(metaStore.get(CONTEXT_KEY));
            if (storedContextId && storedContextId !== contextId) entryStore.clear();
            metaStore.put(contextId, CONTEXT_KEY);
            for (const entry of entries) entryStore.put(entry);
            await done;
        } finally {
            database.close();
        }
        await trim();
    }

    async function removeOldest(count) {
        if (!enabled || count <= 0) return;
        const database = await open();
        try {
            const transaction = database.transaction(ENTRY_STORE, "readwrite");
            const done = transactionComplete(transaction);
            let remaining = count;
            await new Promise((resolve, reject) => {
                const request = transaction.objectStore(ENTRY_STORE).index("createdAt").openCursor();
                request.onerror = () => reject(request.error);
                request.onsuccess = () => {
                    const cursor = request.result;
                    if (!cursor || remaining <= 0) {
                        resolve();
                        return;
                    }
                    cursor.delete();
                    remaining -= 1;
                    cursor.continue();
                };
            });
            await done;
        } finally {
            database.close();
        }
    }

    return { load, clear, appendTurn, removeOldest };
}
