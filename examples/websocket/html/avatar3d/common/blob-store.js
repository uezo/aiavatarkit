export function createBlobStore({
    enabled = true,
    databaseName,
    databaseVersion = 1,
    storeName = "blobs",
}) {
    if (!databaseName) {
        throw new Error("persistence.databaseName is required");
    }

    function open() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(databaseName, databaseVersion);
            request.onupgradeneeded = () => {
                if (!request.result.objectStoreNames.contains(storeName)) {
                    request.result.createObjectStore(storeName);
                }
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    return {
        async get(key) {
            if (!enabled) return null;
            const database = await open();
            return new Promise((resolve, reject) => {
                const transaction = database.transaction(storeName, "readonly");
                const request = transaction.objectStore(storeName).get(key);
                request.onsuccess = () => resolve(request.result ?? null);
                request.onerror = () => reject(request.error);
                transaction.oncomplete = () => database.close();
            });
        },

        async put(key, value) {
            if (!enabled) return;
            const database = await open();
            return new Promise((resolve, reject) => {
                const transaction = database.transaction(storeName, "readwrite");
                transaction.objectStore(storeName).put(value, key);
                transaction.oncomplete = () => {
                    database.close();
                    resolve();
                };
                transaction.onerror = () => {
                    database.close();
                    reject(transaction.error);
                };
            });
        },

        async delete(key) {
            if (!enabled) return;
            const database = await open();
            return new Promise((resolve, reject) => {
                const transaction = database.transaction(storeName, "readwrite");
                transaction.objectStore(storeName).delete(key);
                transaction.oncomplete = () => {
                    database.close();
                    resolve();
                };
                transaction.onerror = () => {
                    database.close();
                    reject(transaction.error);
                };
            });
        },
    };
}
